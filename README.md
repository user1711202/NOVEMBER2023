# NOVEMBER2023

## Abstract

This project explores the application of Self-Supervised Learning (SSL) methods in whole-image classification tasks, divided into object-centric and scene-centric approaches. We utilize low-data variants of ImageNet and Places datasets to evaluate these methods in both transfer and single-dataset setups.

## Experiments and Methodology
#Experiment Types
Object-Centric Classification: Classifying images by specific objects using ImageNet-100 for pretraining, and COCO and STL-10 for the transfer setup.
Scene-Centric Classification: Classifying images by overall scenes using Places-100 for pretraining, and ADE20K and SUN Database for the transfer setup.

SSL Methods Evaluation
The study evaluates four different SSL methods under two distinct setups:

Transfer Setup: SSL pretraining on one dataset and supervised finetuning on another.
Single-Dataset Setup: Both SSL pretraining and supervised finetuning on the same dataset.
Comparative Analysis
We compare SSL methods against deep neural networks (DNNs) pretrained on both the full and low-data versions of ImageNet and Places datasets. The effectiveness of SSL is also measured against DNNs trained from scratch.

