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


