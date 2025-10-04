import os
import sys
import warnings
import logging

from absl import app
from absl import flags
from clu import metric_writers
from clu import periodic_actions
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

import data_providers
from deepvariant import dv_utils
import keras_modeling
from official.modeling import optimization
from pathlib import Path


tf.keras.backend.clear_session()

_LEADER = flags.DEFINE_string(
    'leader',
    'local',
    (
        'The leader flag specifies the host-controller. Possible values: '
        '(1) local=runs locally. If GPUs are available they will be detected'
        ' and used.'
    ),
)

_STRATEGY = flags.DEFINE_enum(
    'strategy',
    'mirrored',
    ['tpu', 'mirrored'],
    'The strategy to use.',
)

_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    '',
    (
        'The directory where the model weights, training/tuning summaries, '
        'and backup information are stored.'
    ),
)

_LIMIT = flags.DEFINE_integer(
    'limit', None, 'Limit the number of steps used for train/eval.'
)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'Run tensorflow eagerly in debug mode.'
)

_DETERMINISTIC_SERIALIZATION = flags.DEFINE_bool(
    'deterministic_serialization',
    True,
    'If True, the saved protos will be '
)

config_flags.DEFINE_config_file('config', "configs/dv_config.py:base")

FLAGS = flags.FLAGS


def train(config: ml_collections.ConfigDict, trial):

  def format_metrics(step, epoch, metrics: dict):
    parts = [f"[{step}] epoch={epoch}"]
    for k, v in metrics.items():
        if hasattr(v, "numpy"):  # tf.Tensor → float
            v = v.numpy()
        parts.append(f"{k}={v}")
    return ", ".join(parts)

  logging.info('Running with debug=%s', FLAGS.debug)
  tf.config.run_functions_eagerly(FLAGS.debug)
  if FLAGS.debug:
    tf.data.experimental.enable_debug_mode()

  logging.info("=== Training Configuration ===")
  logging.info(str(config))
  
  experiment_dir = config.experiment_dir
  logging.info('experiment_dir: %s', experiment_dir)

  model_dir = f'{experiment_dir}/checkpoints'

  logging.info(
      'Use TPU at %s', FLAGS.leader if FLAGS.leader is not None else 'local'
  )
  
  if FLAGS.strategy == 'tpu':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.leader
    )
    tf.config.experimental_connect_to_cluster(resolver, protocol='grpc+loas')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  elif FLAGS.strategy in ['mirrored']:
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise ValueError(f'Unknown strategy: {FLAGS.strategy}')

  # Load config
  train_dataset_config = data_providers.read_dataset_config(
      config.train_dataset_pbtxt
  )

  tune_dataset_config = data_providers.read_dataset_config(
      config.tune_dataset_pbtxt
  )

  input_shape = dv_utils.get_shape_from_examples_path(
      train_dataset_config.tfrecord_path
  )

  # Copy example_info.json to checkpoint path.
  example_info_json_path = os.path.join(
      os.path.dirname(train_dataset_config.tfrecord_path), 'example_info.json'
  )
  if not tf.io.gfile.exists(example_info_json_path):
    raise FileNotFoundError(example_info_json_path)
  tf.io.gfile.makedirs(experiment_dir)
  tf.io.gfile.copy(
      example_info_json_path,
      os.path.join(experiment_dir, 'example_info.json'),
      overwrite=True,
  )

  steps_per_epoch = train_dataset_config.num_examples // config.batch_size
  steps_per_tune = (
      config.num_validation_examples
      or tune_dataset_config.num_examples // config.batch_size
  )

  if FLAGS.limit:
    steps_per_epoch = FLAGS.limit
    steps_per_tune = FLAGS.limit

  # =========== #
  # Setup Model #
  # =========== #

  with strategy.scope():

    # Define Model.
    print('---CHECKPOINT: ', config.init_checkpoint, ' ---')
    model = keras_modeling.inceptionv3(
         input_shape=input_shape,
         weights=config.init_checkpoint,
         init_backbone_with_imagenet=config.init_backbone_with_imagenet,
         config=config,
    )

    # Define Loss Function.
    loss_function = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config.label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def compute_loss(probabilities, labels):
      per_example_loss = loss_function(y_pred=probabilities, y_true=labels)
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=config.batch_size
      )

    decay_steps = int(
        steps_per_epoch * config.learning_rate_num_epochs_per_decay
    )

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=decay_steps,
        decay_rate=config.learning_rate_decay_rate,
        staircase=True,
    )

    logging.info(
        'Exponential Decay:'
        ' initial_learning_rate=%s\n'
        ' decay_steps=%s\n'
        ' learning_rate_decay_rate=%s',
        config.learning_rate,
        decay_steps,
        config.learning_rate_decay_rate,
    )

    if config.warmup_steps > 0:
      warmup_learning_rate = config.learning_rate / 10
      logging.info(
          'Use LinearWarmup: \n warmup_steps=%s\n warmup_learning_rate=%s',
          config.warmup_steps,
          warmup_learning_rate,
      )
      learning_rate = optimization.LinearWarmup(
          warmup_learning_rate=warmup_learning_rate,
          after_warmup_lr_sched=learning_rate,
          warmup_steps=config.warmup_steps,
      )

    # Define Optimizer.
    if config.optimizer == 'adam':
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=learning_rate,
          beta_1=config.beta_1,
          beta_2=config.beta_2,
          epsilon=config.epsilon,
      )
    elif config.optimizer == 'adamw':
      optimizer = tf.keras.optimizers.experimental.AdamW(
          learning_rate=learning_rate,
          beta_1=config.beta_1,
          beta_2=config.beta_2,
          epsilon=config.epsilon,
          weight_decay=config.adam_weight_decay,  
      )
    elif config.optimizer == 'nadam':
      optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    elif config.optimizer == 'rmsprop':
      optimizer = tf.keras.optimizers.RMSprop(
          learning_rate=learning_rate,
          rho=config.rho,
          momentum=config.momentum,
          epsilon=config.epsilon,
      )
    else:
      raise ValueError(f'Unknown optimizer: {config.optimizer}')

  # ================= #
  # Setup Checkpoint  #
  # ================= #

  ckpt_manager = keras_modeling.create_state(
      config,
      model_dir,
      model,
      optimizer,
      strategy,
  )
  print(ckpt_manager)
  state = ckpt_manager.checkpoint

  @tf.function
  def run_train_step(inputs):
    model_inputs, labels = inputs
    with tf.GradientTape() as tape:
      logits = model(model_inputs, training=True)
      probabilities = tf.nn.softmax(logits)
      train_loss = compute_loss(probabilities=probabilities, labels=labels)

    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for metric in state.train_metrics[:-1]:
      metric.update_state(
          y_pred=probabilities,
          y_true=labels,
      )
    state.train_metrics[-1].update_state(train_loss)
    
    # Update learning rate metric
    current_lr = optimizer.learning_rate
    if isinstance(current_lr, tf.distribute.DistributedValues):
      current_lr = current_lr.values[0]  # Get first replica's LR
    lr_metric.update_state(current_lr)
    
    return train_loss

  @tf.function
  def run_tune_step(tune_inputs):
    """Single non-distributed tune step."""
    model_inputs, labels = tune_inputs
    logits = model(model_inputs, training=False)
    probabilities = tf.nn.softmax(logits)
    tune_loss = compute_loss(probabilities=probabilities, labels=labels)

    for metric in state.tune_metrics[:-1]:
      metric.update_state(
          y_pred=probabilities,
          y_true=labels,
      )
      state.tune_metrics[-1].update_state(tune_loss)
    return tune_loss

  @tf.function
  def distributed_train_step(iterator):
    per_replica_losses = strategy.run(run_train_step, args=(next(iterator),))
    state.global_step.assign_add(1)
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )

  @tf.function
  def distributed_tune_step(iterator):
    per_replica_losses = strategy.run(run_tune_step, args=(next(iterator),))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )

  # ============== #
  # Setup Datasets #
  # ============== #
  train_ds = data_providers.input_fn(
      train_dataset_config.tfrecord_path,
      mode='train',
      strategy=strategy,
      n_epochs=config.num_epochs,
      config=config,
      limit=FLAGS.limit,
  )
  tune_ds = data_providers.input_fn(
      tune_dataset_config.tfrecord_path,
      mode='tune',
      strategy=strategy,
      config=config,
      limit=steps_per_tune,
  )

  train_iter, tune_iter = iter(train_ds), iter(tune_ds)
  num_train_steps = steps_per_epoch * config.num_epochs

  logging.info(
      (
          '\n\n'
          'Training Examples: %s\n'
          'Batch Size: %s\n'
          'Epochs: %s\n'
          'Steps per epoch: %s\n'
          'Steps per tune: %s\n'
          'Num train steps: %s\n'
          '\n'
      ),
      train_dataset_config.num_examples,
      config.batch_size,
      config.num_epochs,
      steps_per_epoch,
      steps_per_tune,
      num_train_steps,
  )
  
  # Log learning rate schedule information
  logging.info(
      (
          'Learning Rate Schedule:\n'
          '  Initial LR: %s\n'
          '  Decay steps: %s\n'
          '  Decay rate: %s\n'
          '  Warmup steps: %s\n'
          '  Optimizer: %s\n'
      ),
      config.learning_rate,
      decay_steps,
      config.learning_rate_decay_rate,
      config.warmup_steps,
      config.optimizer,
  )

  # ============= #
  # Training Loop #
  # ============= #

  logging.info("=== Starting Training Loop ===")
  logging.info(f"Total training steps: {num_train_steps}")
  logging.info(f"Steps per epoch: {steps_per_epoch}")
  logging.info(f"Number of epochs: {config.num_epochs}")
 
  metric_writer = metric_writers.create_default_writer(logdir=experiment_dir)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps,
      writer=metric_writer,
      every_secs=300,
      on_steps=[0, num_train_steps - 1],
  )

  # Track the best metric for Optuna
  best_checkpoint_metric_value = float('-inf')
  
  # Create learning rate metric for tracking
  lr_metric = tf.keras.metrics.Mean(name='learning_rate')

  with strategy.scope(): # on all devices

    # Have questions
    def get_checkpoint_metric():
      """Returns the metric we are optimizing for."""
      best_checkpoint_metric_idx = [
          f'tune/{x.name}' for x in state.tune_metrics
      ].index(config.best_checkpoint_metric)
      return state.tune_metrics[best_checkpoint_metric_idx].result().numpy()
    

    def get_current_learning_rate():
      """Returns the current learning rate value."""
      current_lr = optimizer.learning_rate
      if isinstance(current_lr, tf.distribute.DistributedValues):
        current_lr = current_lr.values[0]
        logging.info(f"Learning Rate is instance (DistributedValues): {current_lr}")
      elif callable(current_lr):
        current_lr = current_lr(train_step)
        logging.info(f"Learning Rate is callable: {current_lr}")
      return current_lr.numpy()

    best_checkpoint_metric_value = get_checkpoint_metric()
    print(f"Best checkpoint metric value: {get_checkpoint_metric()}")

    with metric_writers.ensure_flushes(metric_writer):

      def run_tune(train_step, epoch, steps_per_tune):
        logging.info('Running tune at step=%d epoch=%d', train_step, epoch)
        for loop_tune_step in range(steps_per_tune):
          tune_step = loop_tune_step + (epoch * steps_per_tune)
          with tf.profiler.experimental.Trace('tune', step_num=tune_step, _r=1):
            if loop_tune_step % config.log_every_steps == 0:
              logging.info(
                  'Tune step %s / %s (%s%%)',
                  loop_tune_step,
                  steps_per_tune,
                  round(float(loop_tune_step) / float(steps_per_tune), 1)
                  * 100.0,
              )
            distributed_tune_step(tune_iter)

        # Get current learning rate for validation logging
        current_lr = get_current_learning_rate()

        tune_metrics = {f'tune/{x.name}': x.result() for x in state.tune_metrics}
        tune_metrics['tune/learning_rate'] = current_lr
        
        #metric_writer.write_scalars(
        #    train_step,
        #    tune_metrics,
        #)
        logging.info(format_metrics(train_step, epoch, tune_metrics))

        trial.report(tune_metrics['tune/f1_weighted'], epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

      for train_step in range(state.global_step.numpy(), num_train_steps):
        # Calculate current epoch
        epoch = train_step // steps_per_epoch
        if train_step % steps_per_epoch == 0:
          logging.info('=== Starting Epoch %s ===', epoch)
          logging.info('Epoch %s: Training step %d of %d', epoch, train_step, num_train_steps)

        # If we are warmstarting, establish an initial best_checkpoint_metric
        # value before beginning any training.
        if train_step == 0 and (
            config.init_checkpoint or config.init_backbone_with_imagenet
        ):
          logging.info('Performing initial evaluation of warmstart model.')
          run_tune(train_step, epoch, steps_per_tune)
          best_checkpoint_metric_value = get_checkpoint_metric()
          
          # Log initial learning rate TODO need to check
          initial_lr = get_current_learning_rate()
          
          logging.info(
              'Warmstart checkpoint best checkpoint metric: %s=%s, initial LR: %s',
              config.best_checkpoint_metric,
              best_checkpoint_metric_value,
              initial_lr,
          )
          # Reset tune metrics
          for metric in state.tune_metrics:
            metric.reset_states()

        # ===== #
        # train #
        # ===== #

        # Calculate full train step.
        is_last_step = train_step == (num_train_steps - 1)

        with tf.profiler.experimental.Trace('train', step_num=train_step, _r=1):
          distributed_train_step(train_iter)

        # Log metrics
        report_progress(train_step)

        if (train_step % config.log_every_steps == 0) or is_last_step:
          metrics_to_write = {
              f'train/{x.name}': x.result() for x in state.train_metrics
          }
          
          # Add learning rate from metric (more accurate)
          metrics_to_write['train/learning_rate'] = lr_metric.result().numpy()
          metrics_to_write['epoch'] = epoch
          
          #metric_writer.write_scalars(
          #    train_step,
          #    metrics_to_write,
          #)
          logging.info(format_metrics(train_step, epoch, metrics_to_write))

          # Reset train metrics and learning rate metric
          for metric in state.train_metrics:
            metric.reset_states()
          lr_metric.reset_states()

        # ==== #
        # tune #
        # ==== #
        # Run tune at every epoch, periodically, and at final step.
        if (
            (train_step > 0 and train_step % steps_per_epoch == 0)
            or (train_step > 0 and train_step % config.tune_every_steps == 0)
            or is_last_step
        ):
          run_tune(train_step, epoch, steps_per_tune)

          current_metric = get_checkpoint_metric()
          if current_metric > best_checkpoint_metric_value:
            best_checkpoint_metric_value = current_metric
            checkpoint_path = ckpt_manager.save(train_step)
            # Reset early stopping counter
            state.early_stopping.assign(0)
            logging.info(
                'Saved checkpoint %s=%s step=%s epoch=%s path=%s',
                config.best_checkpoint_metric,
                get_checkpoint_metric(),
                train_step,
                epoch,
                checkpoint_path,
            )
          else:
            if (
                config.early_stopping_patience
                and state.early_stopping.value()
                >= config.early_stopping_patience
            ):
              break
            logging.info(
                'Skipping checkpoint with %s=%s < previous best %s=%s',
                config.best_checkpoint_metric,
                get_checkpoint_metric(),
                config.best_checkpoint_metric,
                best_checkpoint_metric_value,
            )
            state.early_stopping.assign_add(1)
          if config.early_stopping_patience:
            
            logging.info(f"tune/early_stopping: {state.early_stopping.value()}")
            #metric_writer.write_scalars(
            #    train_step,
            #    {'tune/early_stopping': state.early_stopping.value()},
            #)
            #logging.info(format_metrics(train_step, epoch, metrics_to_write))
            
          # Reset tune metrics
          for metric in state.tune_metrics:
            metric.reset_states()

    # After training completes, load the latest checkpoint and create
    # a saved model (.pb) and keras model formats.
    checkpoint_path = ckpt_manager.latest_checkpoint
    print('ckpt_path: ',checkpoint_path)
    
    if not checkpoint_path:
      logging.info('No checkpoint found.')
      return best_checkpoint_metric_value

    # The latest checkpoint will be the best performing checkpoint.
    logging.info('Loading best checkpoint: %s', checkpoint_path)
    tf.train.Checkpoint(model).restore(checkpoint_path).expect_partial()

    logging.info('Saving model using saved_model format.')
    saved_model_dir = checkpoint_path
    model.save(saved_model_dir, save_format='tf')
    # Copy example_info.json to saved_model directory.
    tf.io.gfile.copy(
        example_info_json_path,
        os.path.join(
            saved_model_dir,
            'example_info.json',
        ),
        overwrite=True,
    )
    print('---finish_good---')
    metric_writer.close()

    return best_checkpoint_metric_value

def main(unused_argv):
  keep_running = True
  while keep_running:
    try:
      best_metric = train(FLAGS.config)
      print('---end_train---')
      print(f'Best metric achieved: {best_metric}')
      keep_running = False  # Training completed successfully.
    except tf.errors.UnavailableError as error:
      logging.warning(
          'UnavailableError encountered during training: %s.', error
      )
      print('!exept!')
  return

if __name__ == '__main__':
    sys.argv = sys.argv[:1]
    #logging.set_verbosity(logging.INFO)
    warnings.filterwarnings(
         'ignore', module='tensorflow_addons.optimizers.average_wrapper')
    try:
        app.run(main)
    except SystemExit:
        print('!sys exit!')
