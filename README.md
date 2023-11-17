# Evaluating SSL Methods in Whole-Image Classification

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


## Installation Instructions
To install and set up the project...

## Usage
To use this project for image classification...

## Results

### Top-5 Accuracy for the Single-Dataset Setup

ImageNet-100/Places-100 has been utilized both for pretraining and downstream finetuning in the object-centric/scene-centric cases, respectively. The overall best accuracy per downstream dataset is highlighted in bold; if it is from a non-SSL approach, then the best SSL method's accuracy is underlined.

| Pretraining        | IN-100      | Places-100  |
| ------------------ | ----------- | ----------- |
| SimCLR (ResNet)    | **73.5%**   | *69.8%*     |
| DINO (ViT)         | 72.1%       | 68.5%       |
| MAE (ViT)          | 72.7%       | 69.1%       |
| DeepClusterV2 (ResNet) | 70.3%   | 66.8%       |
| None/Random (ResNet)   | 70.6%   | 70.3%       |
| None/Random (ViT)      | 71.2%   | **70.9%**   |

## Accuracy Metrics for the Transfer Setup

ImageNet-100/Places-100 has been utilized for pretraining in the object-centric/scene-centric cases, respectively, unless otherwise explicitly specified. Top-1/top-5 accuracy is reported for STL-10/all other cases, respectively. The overall best accuracy per downstream dataset is highlighted in bold; if it is from a non-SSL approach, then the best SSL method's accuracy is underlined.

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


