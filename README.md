# Alternative Neural Networks for DeepVariant
[Homepage](https://agurianova.github.io/deepvariant-alternative-models/) | [Paper]()

This repository provides a workflow to 
- train alternative models for DeepVariant
- run whole DeepVariant pipeline with implemented custom models

## GPU Setup with Docker

Build Docker image and run GPU-enabled container based on the official google/deepvariant:1.6.1-gpu image. Then, open **Jupyter environment** in your browser using tokenized URL printed in the container logs.

```bash
docker build -t deepvariant:v1 .
docker run \
   -v /src/:/opt/deepvariant/src \
   -v /data/:/opt/deepvariant/data \
   --gpus all \
   -p 1234:8888 \
   deepvariant:v1
```

## Alternative Model Training

From a Jupyter terminal or a shell inside the container, launch model training with `train.py`.

To configure the run, either modify `dv_config.py`, or override individual parameters inline using `--config.*` flags. You can choose different model architectures, specify datasets, and adjust batch size, learning rate, logging frequency, and more.

Support for `EfficientNet` models is now available, we encourage you to explore them or add your own custom model.
<details>
<summary>List of supported models:</summary>

| Model           | model_type           |
|-----------------|---------------------|
| EfficientNet-B0  | 'efficientnetb0'    |
| EfficientNet-B1  | 'efficientnetb1'    |
| EfficientNet-B2  | 'efficientnetb2'    |
| EfficientNet-B3  | 'efficientnetb3'    |
| EfficientNet-B4  | 'efficientnetb4'    |
| EfficientNet-B5  | 'efficientnetb5'    |
| EfficientNet-B6  | 'efficientnetb6'    |
| EfficientNet-B7  | 'efficientnetb7'    |

</details>

```bash
python train.py --config=dv_config.py:base \
   --config.model_type='efficientnetb3' \
   --config.batch_size=128 \
   --config.num_epochs=10 \
   --config.learning_rate=1e-3
```
Results are stored as:

```
current-run
└──training/
   ├── checkpoints/     # Trained model weights and states
   └── logs.log         # Detailed training metrics and logs
```

## Custom Model Integration and Testing

You can now run and test full DeepVariant pipeline using recently trained model with `test.sh`. Provide `current-run` directory; model details will be automatically loaded from `training/checkpoints/`. 
You can also specify test dataset or restrict evaluation to specific chromosomes or regions. Variant call results will be saved to a VCF file, and `hap.py` testing will be launched automatically. 

```bash
docker pull jmcdani20/hap.py:v0.3.12 # Pull hap.py image if not already available
sh test.sh current-run
```

Results are stored as:

```
current-run
└──testing/
   ├── test.vcf.gz              # Variant call results
   └── happy.output.summary     # Final evaluation metrics
```

## Multi-Trial Training and Testing

You can orchestrate **multiple training/testing runs** using [Optuna](https://optuna.org/) to:
- Automatically find **optimal hyperparameters** for training
- Perform **cross-validation** with multiple runs

Custom `optuna_train.py` script runs `train()` multiple times using parameters suggested by Optuna’s sampling strategy (e.g., different learning rates, optimizers, seeds, etc). Each trial is saved to a separate subdirectory under the `current-run`.

Example command to run hyperparameter tuning:
```bash
python optuna_train.py
```
Results are stored as:
```
current-run
├── optuna_trial_0/
│   ├── training/
│   └── testing/
├── optuna_trial_1/
└── optuna_trial_2/
```   
----
Feel free to explore and experiment, your feedback and contributions are always welcome!