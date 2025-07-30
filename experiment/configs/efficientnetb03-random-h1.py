import ml_collections

def get_config(config_name: str) -> ml_collections.ConfigDict:
  """Training parameters."""
  config = ml_collections.ConfigDict()

  config.model_type = 'efficientnetb03'
  config.trial = 0  # Used to allow for replicates during training.

  # Default Dataset
    
  config.train_dataset_pbtxt = '/opt/deepvariant/data/pbtxts/HG0015_30x_train_shuffled.pbtxt'
  config.tune_dataset_pbtxt = '/opt/deepvariant/data/pbtxts/HG0015_30x_eval_shuffled.pbtxt'
  config.experiment_dir = '/opt/deepvariant/src/architecture_change/experiment/results/efficientnetb03-random-h1/'
  config.log_file = '/opt/deepvariant/src/architecture_change/experiment/results/efficientnetb03-random-h1/logs.log'

  config.best_checkpoint_metric = 'tune/f1_weighted'
  config.batch_size = 128 #1024 #16384
  config.num_epochs = 5
  config.num_validation_examples = 0 #5000 #1500000
  config.optimizer = 'rmsprop'

  # Training hyperparameters
  config.learning_rate = 1e-3
  config.learning_rate_num_epochs_per_decay = 2.0
  config.learning_rate_decay_rate = 0.947
  config.average_decay = 0.999
  config.label_smoothing = 1e-6
  config.rho = 0.9
  config.momentum = 0.9
  config.epsilon = 1.0
  config.warmup_steps = 10_000
  config.init_checkpoint = None #'/opt/deepvariant/src/dv_ckpt_1.6.1/deepvariant.wgs.ckpt'
  config.init_backbone_with_imagenet = False
  config.best_metrics = 'tune/f1_weighted'
  config.weight_decay = 0.00004
  config.backbone_dropout_rate = 0.2

  # Stop training when this many consecutive evaluations yield no improvement.
  config.early_stopping_patience = 100

  # TensorBoard Options
  config.log_every_steps = 100
  # Tuning happens at every epoch. The frequency can be increased here.
  config.tune_every_steps = 5000

  # Data Pipeline Options
  config.prefetch_buffer_bytes = 16 * 1000 * 1000
  config.shuffle_buffer_elements = 10_000
  config.input_read_threads = 8

  # Augmentation
  config.augmentations = [] #[auglib.readbase, auglib.strand] # # #[]
  config.aug_probability = [] #[1.0] #[0.5, 0.5] #[0.5] # #[]
  config.snp_important = [False]

  # Placeholder value for limiting training examples. 0=No limit.
  config.limit = 0

  if config_name == 'base':
    # Use the base config.
    pass
  else:
    raise ValueError(f'Unknown config_name: {config_name}')

  return config