# Self-supervised visual learning in the low-data regime: a comparative evaluation

## Abstract

This project explores the application of Self-Supervised Learning (SSL) methods in whole-image classification tasks, divided into object-centric and scene-centric approaches. We utilize low-data variants of ImageNet and Places datasets to evaluate these methods in both transfer and single-dataset setups.

## Experiments and Methodology
Object-Centric Classification: Classifying images by specific objects using ImageNet-100 for pretraining, and COCO and STL-10 for the transfer setup.
Scene-Centric Classification: Classifying images by overall scenes using Places-100 for pretraining, and ADE20K and SUN Database for the transfer setup.

## SSL Methods Evaluation
The study evaluates four different SSL methods under two distinct setups:
Transfer Setup: SSL pretraining on one dataset and supervised finetuning on another.
Single-Dataset Setup: Both SSL pretraining and supervised finetuning on the same dataset.
Comparative Analysis
We compare SSL methods against deep neural networks (DNNs) pretrained on both the full and low-data versions of ImageNet and Places datasets. The effectiveness of SSL is also measured against DNNs trained from scratch.

### Robustness Evaluation
Also a set of complementary experiments was conducted to assess the robustness of the four SSL methods:

#### Noisy-ImageNet-100
- **Objective**: Evaluate resilience against various forms of noise and corruptions.
- **Datasets**: 
  - ImageNet-A, ImageNet-P, and ImageNet-C, adapted to ImageNet-100's class scope.
- **Approach**: 
  - The ImageNet-100-pretrained variants of the SSL methods, after regular downstream finetuning, were tested on these adapted noisy datasets.

#### Imbalanced-ImageNet-100
- **Objective**: Evaluate resilience against uneven class distribution.
- **Dataset**: 
  - An imbalanced variant of ImageNet, named ImageNet-Long-Tail, adapted for ImageNet-100.
- **Approach**: 
  - The ImageNet-100-pretrained SSL methods underwent supervised downstream finetuning on this imbalanced dataset.
- **Comparisons**: 
  - The SSL methods were compared against supervised training/pretraining on ImageNet-1k and ImageNet-100, as well as direct downstream training from scratch.

### Cross-Domain Robustness
- **Benchmark**: Visual Task Adaptation Benchmark (VTAB).
- **Approach**: 
  - The ImageNet-100-pretrained SSL variants were finetuned and tested across the 19 VTAB downstream datasets.
- **Task Groups**: 
  - The benchmark includes "natural," "specialized," and "structured" task groups.

### Domain-Specific Experiments
- **Objective**: Evaluate the performance of SSL methods in specialized image domains.
- **Domains**:
  - **Remote Sensing**: Using MLRSNet for pretraining and AID for downstream.
  - **Medical Imaging**: Using MedPix for pretraining and ChestX-Det for downstream.
  - **Security Imaging**: Using subsets of the SIXRay dataset for both pretraining and downstream tasks.
- **Approach**: 
  - In-domain SSL pretraining was compared against supervised pretraining on ImageNet-100 and ImageNet-1k, along with direct downstream training from scratch.
- **Note**: All domain-specific datasets used are low-data, following the article's criteria.


## Requirements
torch
torchvision
tqdm
einops
wandb
pytorch-lightning
lightning-bolts
torchmetrics
scipy
timm

## Usage
for training python3 main_pretrain.py \
    # path to training script folder
    --config-path scripts/pretrain/imagenet-100/ \
    # training config name
    --config-name barlow.yaml
    # add new arguments (e.g. those not defined in the yaml files)
    # by doing ++new_argument=VALUE
    # pytorch lightning's arguments can be added here as well.

   for offline linear evaluation, follow the examples in scripts/linear or scripts/finetune for finetuning the whole backbone.
   optimal hyparparameters on yamls files

## Results

### Top-5 Accuracy for the Single-Dataset Setup



| Pretraining        | IN-100      | Places-100  |
| ------------------ | ----------- | ----------- |
| SimCLR (ResNet)    | **73.5%**   | *69.8%*     |
| DINO (ViT)         | 72.1%       | 68.5%       |
| MAE (ViT)          | 72.7%       | 69.1%       |
| DeepClusterV2 (ResNet) | 70.3%   | 66.8%       |
| None/Random (ResNet)   | 70.6%   | 70.3%       |
| None/Random (ViT)      | 71.2%   | **70.9%**   |

## Accuracy Metrics for the Transfer Setup


| Pretraining           | COCO (Object-centric) | STL-10 (Object-centric) | ADE20K (Scene-centric) | SUN DB (Scene-centric) |
| --------------------- | --------------------- | ----------------------- | ---------------------- | ---------------------- |
| SimCLR (ResNet)       | 70.8%                 | 64.2%                   | 49.4%                  | 51.7%                  |
| DINO (ViT)            | *75.7%*               | *67.1%*                 | *53.1%*                | *55.2%*                |
| MAE (ViT)             | 74.4%                 | 65.7%                   | 51.1%                  | 53.3%                  |
| DeepClusterV2 (ResNet)| 67.0%                 | 59.3%                   | 46.2%                  | 48.1%                  |
| Supervised (ResNet)   | 73.2%                 | 67.8%                   | 54.3%                  | 53.5%                  |
| Supervised (ViT)      | 74.9%                 | 69.7%                   | 56.2%                  | 55.8%                  |
| Supervised-Large (ResNet) | 79.1% (IN-1k pretr.) | 73.6% (IN-1k pretr.) | 57.4% (Places-205 pretr.) | 59.4% (Places-205 pretr.) |
| Supervised-Large (ViT) | **80.4%** (IN-1k pretr.) | **75.4%** (IN-1k pretr.) | **58.9%** (Places-205 pretr.) | **61.2%** (Places-205 pretr.) |

## Accuracy Metrics for Noisy-ImageNet-100 (Top-5)


| Pretraining          | IN-A-100     | IN-P-100     | IN-C-100     |
| -------------------- | ------------ | ------------ | ------------ |
| SimCLR (ResNet)      | 70.3%        | 75.1%        | 68.3%        |
| DINO (ViT)           | 73.0%        | *78.2%*      | *72.6%*      |
| MAE (ViT)            | *74.6%*      | 76.5%        | 70.4%        |
| DeepClusterV2 (ResNet)| 66.4%       | 68.7%        | 64.2%        |
| Supervised (ResNet)  | 77.2%        | 78.0%        | 73.5%        |
| Supervised-Large (ResNet) | 81.8%   | 83.0%        | 76.5%        |
| Supervised (ViT)     | 78.3%        | 78.5%        | 74.0%        |
| Supervised-Large (ViT)| **82.5%**   | **84.2%**    | **77.6%**    |

## Accuracy Metrics for Imbalanced-ImageNet-100 and VTAB



| Pretraining          | Imbalanced-IN-100 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
| -------------------- | ----------------- | ------------ | ---------------- | --------------- |
| SimCLR (ResNet)      | 63.2%             | 62.5%        | 58.3%            | 60.1%           |
| DINO (ViT)           | *79.1%*           | 65.2%        | 61.8%            | 63.0%           |
| MAE (ViT)            | 69.3%             | *66.7%*      | *62.9%*          | *64.5%*         |
| DeepClusterV2 (ResNet)| 59.1%            | 60.3%        | 56.7%            | 58.2%           |
| Supervised (ResNet)  | 78.2%             | 63.2%        | 62.8%            | 62.5%           |
| Supervised-Large (ResNet)| 85.3%         | 66.9%        | 63.4%            | 65.8%           |
| Supervised (ViT)     | 80.0%             | 67.1%        | **64.7%**        | 66.0%           |
| Supervised-Large (ViT)| **86.2%**        | **68.2%**    | 64.2%            | **66.5%**       |


## Top-1 Accuracy for Domain-Specific Setups


| Pretraining                  | AID            | ChestX-Det     | SIXRay-10      |
| ---------------------------- | -------------- | -------------- | -------------- |
| SimCLR (ResNet)              | 58.1%          | 68.2%          | 73.3%          |
| DINO (ViT)                   | *65.7%*        | **76.4%**      | **74.1%**      |
| MAE (ViT)                    | 63.5%          | 72.1%          | 69.7%          |
| DeepClusterV2 (ResNet)       | 57.4%          | 70.6%          | 66.3%          |
| Supervised (IN-100, ResNet)  | 60.2%          | 70.3%          | 68.4%          |
| Supervised-Large (IN-1k, ResNet) | 66.2%     | 72.5%          | 70.2%          |
| Supervised (IN-100, ViT)     | 65.1%          | 73.6%          | 72.5%          |
| Supervised-Large (IN-1k, ViT) | **68.7%**     | 74.5%          | 73.9%          |




