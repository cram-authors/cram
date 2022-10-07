# CrAM


## About

This repository contains the scripts required for reproducing the results on CIFAR10 and ImageNet, as presented in the ICLR2023 submission.

## Structure

- `main.py` is the main script for launching a training run using CrAM or a different optimizer. In addition to passing the arguments related to dataset and architecture, the training details (type of optimizer, hyperparameters, learning rate scheduler) are passed through a config file. Examples of config files for CrAM training can be found in the `configs/` folder
- `manager.py` contains the Manager class, which trains the model with the desired configs, through the `run` method
- `optimization/` module contains the custom optimizers (Topk-CrAM and SAM), and a custom cosine annealing learning rate scheduler with linear rate warm-up 
- `load_training_configs.py` contains functions for loading the optimizer, learning rate scheduler and training hyperparameters from the config file
- `models/` contains the available types of models; additionally, we use the ResNet ImageNet models as defined in the Torchvision library
- `utils/` contains different utilities scripts (e.g. for loading datasets, saving and loading checkpoints, functions for one-shot pruning or fixing batch norm statistics)
- `generate_calibration_dset.py` creates a calibration set of 1000 Imagenet training samples, and copies them to a different folder; to be used when doing Batch Norm tuning after pruning
- `demo_get_one_shot_pruning_results.py` script that loads a trained CrAM checkpoint, prunes it one-shot to a desired sparsity, does Batch Norm tuning on a calibration set, and checks the validation aaccuracy after each of these operations


## How to run

We provide a few sample bash scripts, to reproduce our results using CrAM+ Multi (sparse grads) on CIFAR10 and ImageNet, as well as CrAM+ k95 on RN18/VGG16 (CIFAR10).

Sample bash scripts:
- CIFAR10: `run_cifar10_resnet20_cram_demo.sh`, `run_cifar10_big_resnet18_cram_plus_k95_sparse_grad.sh`, `run_cifar10_vgg16_cram_plus_k95_sparse_grad.sh` (for RN20/ RN18/ VGG16)
- ImageNet: `run_imagenet_resnet50_cram_demo.sh` (for RN50)

`bash SCRIPT_NAME.sh {GPU id}` (GPU id only needed for CIFAR10 scripts)

To obtain the results after one-shot pruninig + BNT:
- change the paths to the dataset and provide path to trained checkpoint, and run: 
`python demo_get_one_shot_pruning_results.py --use_calib --calib_size 1000`
- for BNT on fixed calibration set (only on ImageNet), first create the calibration set using `generate_calibration_dset.py` and then
`python demo_get_one_shot_pruning_results.py --fixed_calib --use_calib --calib_size 1`

We also use Weights & Biases (Wandb) for tracking our experiments. This can be enabled through '--use_wandb' inside the bash scripts. Not enabling it will use the print function by default.


## Requirements

- python 3.9
- torch 1.8.1 
- torchvision 0.9.1
- wandb 0.12.17

