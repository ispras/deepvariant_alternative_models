import ml_collections

def get_config(config_name: str) -> ml_collections.ConfigDict:
  """Training parameters."""
  config = ml_collections.ConfigDict()

  config.seed = 123

  # Model
  config.model_type = 'inception_v3'
  config.init_checkpoint = None # dv weights 
  config.init_backbone_with_imagenet = False # imagenet weights
  
  # Model Regularization
  config.weight_decay = 0.00004
  config.backbone_dropout_rate = 0.2
  config.label_smoothing = 1e-6
  
  # Experimantal dir and log file
  config.experiment_dir = '/opt/deepvariant/src/current-run/training/'
  config.log_file = '/opt/deepvariant/src/current-run/training/logs.log'

  # Dataset
  config.train_dataset_pbtxt = '/opt/deepvariant/data/pbtxts/training_set_HG001_30x_chr10.pbtxt'
  config.tune_dataset_pbtxt = '/opt/deepvariant/data/pbtxts/validation_set_HG001_30x_chr20.pbtxt'

  # Model Performance Monitoring
  config.best_checkpoint_metric = 'tune/f1_weighted'  # Metric to track for best checkpoint saving

  # Data Handling
  config.batch_size = 128 #16384                     # Number of samples per training batch
  config.num_validation_examples = 0                 # Number of validation samples during training

  # Training Duration
  config.num_epochs = 10

  # Learning Rate Hyperparameters
  config.learning_rate = 1e-3                        # Initial learning rate
  config.learning_rate_num_epochs_per_decay = 2.0    # Epochs per learning rate decay step
  config.learning_rate_decay_rate = 0.947            # Learning rate decay multiplier
  config.warmup_steps = 15                           # Number of warmup steps to gradually increase LR from [initial lr]/10
  
  # Optimizer Hyperparameters
  config.optimizer = 'rmsprop' # default
  # RMSprop parameters
  config.rho = 0.9
  config.momentum = 0.9
  config.epsilon = 1.0
  config.average_decay = 0.999            

  # Adam parameters
  config.beta_1 = 0.9
  config.beta_2 = 0.999
  config.adam_weight_decay = 0.01

  config.best_metrics = 'tune/f1_weighted'

  # Logging
  config.log_every_steps = 5

  # Tune
  config.tune_every_steps = 10_000 
  config.early_stopping_patience = 10_000 # Stop training when this many consecutive evaluations yield no improvement.

  # Data Pipeline Options
  config.prefetch_buffer_bytes = 16 * 1000 * 1000
  config.shuffle_buffer_elements = 10_000
  config.input_read_threads = 8

  # Placeholder values 
  config.limit = 0 # limiting training examples. 0=No limit.
  config.trial = 0  

  if config_name == 'base':
    pass
  else:
    raise ValueError(f'Unknown config_name: {config_name}')

  return config