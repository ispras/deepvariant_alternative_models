import os
import optuna
import tensorflow as tf
from absl import flags
import absl.logging

import logging
import sys
from pathlib import Path
from dv_config import get_config

class TeeStream:
    """Tee-stream for stdout/stderr."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def setup_logger(log_path: str):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # stdout/stderr → tee
    log_file_stream = open(log_path, "a")
    sys.stdout = TeeStream(sys.__stdout__, log_file_stream)
    sys.stderr = TeeStream(sys.__stderr__, log_file_stream)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.__stdout__)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

    absl.logging._warn_preinit_stderr = False
    absl.logging.use_python_logging()

tf.keras.backend.clear_session()

def setup_flags_for_training():
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(['optuna_train.py'])

def objective(trial):
    setup_flags_for_training()
    
    from train import train
    
    config = get_config('base')

    # Trial dir and logs
    config.experiment_dir = f'{config.experiment_dir}optuna_trial_{trial.number}'
    os.makedirs(config.experiment_dir, exist_ok=True)
    config.log_file = f'{config.experiment_dir}/logs.log'

    # Logger
    setup_logger(config.log_file)
    print(f"Optuna. Trial {trial.number}. Logger set up at {config.log_file}")

    #------- Substitute original config with optuna -------#

    # Select epochs
    config.num_epochs = 10

    # Select optimizer
    range_optimizers = ["rmsprop","adamw"]
    config.optimizer = trial.suggest_categorical("optimizer", range_optimizers)
    if config.optimizer == "adamw":
        config.weight_decay = 0
    print(f"Optuna. Trial {trial.number}. Range of optimizers: {range_optimizers}, selected {config.optimizer}")

    # Select learning rate
    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True) 
    print(f"Optuna. Trial {trial.number}. Range of learning rates: 1e-8 .. 1, selected {config.learning_rate}")

    # Select model
    range_models = ["inception_v3","efficientnetb03"]
    config.model_type = trial.suggest_categorical("model_type", range_models)
    print(f"Optuna. Trial {trial.number}. Range of models: {range_models}, selected {config.model_type}")

    # Select seed
    config.seed = trial.suggest_categorical("seed", [0, 1, 12, 123, 13, 14, 145, 134, 42, 99]) # random seeds
    
    # Select other
    #config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    #config.backbone_dropout_rate = trial.suggest_float("backbone_dropout_rate", 0.0, 0.5)
    
    #------- Train -------#
    try:
        print(f"Optuna. Trial {trial.number}. Starting: train(config)")
        best_metric = train(config, trial)
        print(f"Optuna. Trial {trial.number}. ✅ Completed - F1: {best_metric:.4f}")
        return best_metric
    
    except Exception as e:
        print(f"Optuna. Trial {trial.number}. ❌ Failed: {str(e)}")
        return float('-inf')

if __name__ == '__main__':

    # Pruner
    pruner = optuna.pruners.SuccessiveHalvingPruner(
    min_resource=10,
    reduction_factor=3,
    min_early_stopping_rate=0
    )

    # Study
    study = optuna.create_study(
    direction="maximize", 
    sampler=optuna.samplers.TPESampler(seed=123), 
    pruner=optuna.pruners.NopPruner() #pruner 
    )

    # Optimize objective
    study.optimize(
    objective,
    n_trials=10,      
    timeout=None,  
    n_jobs=1  
    )

    print(f"Optuna Report. Best Hyperparameters: {study.best_params}")
    print(f"Optuna Report. Best F1: {study.best_value}")
    print(f"Optuna Report. Best Trial: {study.best_trial.number}")