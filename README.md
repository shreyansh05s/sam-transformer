# Sharpness Aware Transformers for Cost-Efficient Image Classification

## Table of Contents

- [Sharpness Aware Transformers for Cost-Efficient Image Classification](#sharpness-aware-transformers-for-cost-efficient-image-classification)
  - [Overview](#overview)
  - [Installation](#installation)
      - [Additional setup for wandb logging](#additional-setup-for-wandb-logging)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Experiments](#experiments)
    - [VIT](#vit)
    - [VIT-COSINE](#vit-cosine)
    - [VIT-ASAM](#vit-asam)
    - [SWIN-ASAM](#swin-asam)
  - [Benchmarking](#benchmarking)

## Overview

This is a Pytorch implementation of Huggingface models for image classification.
Models include:
- [ViT](https://arxiv.org/abs/2010.11929)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ResNet](https://arxiv.org/abs/1512.03385)

Datasets included:
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

Optimizer:
- [AdamW](https://arxiv.org/abs/1711.05101)
- [SAM](https://arxiv.org/abs/2010.01412)

## Installation

First insure python 3.8.10 or higher is installed.

Then create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

```bash
source venv/bin/activate
```

Install the requirements.

```bash
pip install -r requirements.txt
```

### Additional setup for wandb logging

- If you want to use wandb logging, you will need to create an account at [wandb.ai](https://wandb.ai/).

- Then get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

- Then login to wandb from the command line.

```bash
wandb login
```

## Dataset

The CIFAR100 dataset is included in the `datasets` library. The dataset is downloaded and extracted automatically when the `train.py` or `eval.py` script is run.

## Training

To train a model, we can call the `train.py` script, whose arguments can be found by running the following:

```bash
python train.py --help
```

For example, to train a ViT model on CIFAR100, we can run the following:

```bash
python train.py --model vit --batch_size 32 --epochs 5 --lr 0.01 --optimizer sam
```

For training with wandb logging, we can run the following:

```bash
python train.py --model vit --batch_size 32 --epochs 5 --lr 0.01 --optimizer sam -W
```


## Evaluation

To evaluate a model, we can call the `eval.py` script, whose arguments can be found by running the following:

```bash
python eval.py --help
```

For example, to evaluate a ViT model on CIFAR100, we can run the following:

```bash
python eval.py --model VIT
```

For Evaluation on all models, we can run the following:

```bash
python eval.py
```

## Hyperparameter Tuning

Hyperparameter tuning was done manually by running variations of arguments in the `train.py` script. The best performing hyperparameters were then used for the experiments.

Default hyperparameters can be found in the `train.py` script. And can be seen in the table below:

| Hyperparameter | Default Value |
| -------------- | ------------- |
| batch_size     | 32            |
| epochs         | 5             |
| lr             | 0.01          |
| optimizer      | sam           |
| scheduler      | step          |
| gamma          | 0.95          |
| momentum       | 0.9           |
| step_size      | 100           |

## Experiments

In this section, we will highlight steps to reproduce the experiments in the paper. Run all commands below:

### VIT

To train a ViT model on CIFAR100, we can run the following(also referred to as the VIT-ADAM experiment):

```bash
python train.py --model vit --batch_size 32 --epochs 5 --lr 0.01 --optimizer adam --scheduler step
```

### VIT-COSINE

To train a ViT model with cosine decay on CIFAR100, we can run the following:

```bash
python train.py --model vit --batch_size 32 --epochs 5 --lr 0.01 --optimizer adam --scheduler cosine
```

### VIT-ASAM

To train a ViT model with SAM optimizer on CIFAR100, we can run the following(also referred to as the VIT-STEP experiment):

```bash
python train.py --model vit --batch_size 32 --epochs 5 --lr 0.01 --optimizer sam
```

### SWIN-ASAM

To train a Swin Transformer model with SAM optimizer on CIFAR100, we can run the following(also referred to as the SWIN-ASAM-FROZEN experiment):

```bash
python train.py --model swin --batch_size 32 --epochs 5 --lr 0.01 --optimizer sam --freeze
```

## Benchmarking

To benchmark all models on CIFAR100, we can run the following:

```bash
python eval.py
```

## Notes

- The batch size utilized for experiments is 32. Depending on your GPU memory, you may need to adjust this value.

<!-- cite sam.py -->
## References

```bibtex
@misc{foret2021sharpnessaware,
      title={Sharpness-Aware Minimization for Efficiently Improving Generalization}, 
      author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
      year={2021},
      eprint={2010.01412},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{liu2021swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}, 
      author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
      year={2021},
      eprint={2103.14030},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{dosovitskiy2020image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Known-Working Environment

| Property | Value |
|----------|-------|
| OS | Ubuntu 22.04 / Windows 11 |
| Python | 3.8.10+ (3.10 recommended) |
| PyTorch | 2.0+ |
| CUDA | Optional (GPU recommended for training) |

## Course Context

Seminar Multimedia and Deep Learning — Semester 1, VU Amsterdam AI Master's programme.
