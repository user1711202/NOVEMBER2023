#Datasets Summary

ImageNet-1k (https://ieeexplore.ieee.org/document/5206848) :It contains 1.3 million natural RGB images, uniformly distributed into 1,000 classes and is organized according to the WordNet hierarchy. Each image is assigned only one class label, which is typically object-centric. It is split into a training/validation/test set of 1,281,167/50,000/100,000 images, respectively.
ImageNet-100 (https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.pdf): A low-data variant of ImageNet-1k with 100 classes and 160,000 images. Split into training/validation/test sets of 100,000, 30,000, and 30,000 images, respectively.
Places-365 (https://paperswithcode.com/dataset/places): Contains 10 million images across 434 scene-centric classes. Split into training/validation/test sets of 8 million, 36,000, and 328,000 images, respectively.
Places-100 (https://dl.acm.org/doi/abs/10.1145/3460120.3484571): A low-data variant of Places-365 with 100 classes and 40,000 images. Split into training/test sets of 30,000 and 10,000 images, respectively.
STL-10 (https://cs.stanford.edu/~acoates/stl10/): Designed for unsupervised representation learning, it includes 500 labeled training images, 800 labeled test images, and 100,000 unlabeled images, across 10 classes.
COCO (https://cocodataset.org/#home): Contains 164,000 images annotated for object detection, segmentation, and captioning. Split into training/validation/test sets of 82,000, 41,000, and 41,000 images, respectively.
ADE20K (https://groups.csail.mit.edu/vision/datasets/ADE20K/): Annotated for semantic scene parsing and classification, with 25,574 training and 2,000 test images across 365 classes.
SUN Database (https://groups.csail.mit.edu/vision/SUN/hierarchy.html): Includes 108,754 images across 397 classes. Split into training/validation/test sets of 76,128, 10,875, and 21,750 images, respectively.
ImageNet-A (https://github.com/hendrycks/natural-adv-examples): Comprises 30,000 images from 200 classes of ImageNet-1k, selected as natural adversarial examples. Used only for inference.
ImageNet-P (https://github.com/hendrycks/robustness): A modification of the ImageNet-1k validation set with 200,000 images, perturbed with various transformations. Used only for inference.
ImageNet-C (https://github.com/hendrycks/robustness): Similar to ImageNet-P but with heavier corruptions, containing 200,000 images. Used only for inference.
ImageNet-Long-Tail (https://paperswithcode.com/sota/long-tail-learning-on-imagenet-lt): Features imbalanced class sizes with 115,846 images. Split into training/validation/test sets of 81,092, 17,377, and 17,377 images, respectively.
MLRSNet (https://github.com/cugbrs/MLRSNet): Designed for multi-label remote sensing image classification with 100,000 images. Split into training/validation/test sets of 80,000, 10,000, and 10,000 images, respectively.
AID (https://captain-whu.github.io/AID/): A high-resolution aerial image dataset with 10,000 images for land-use classification. Split into training/validation/test sets of 8,000, 1,000, and 1,000 images, respectively.
MedPix (https://medpix.nlm.nih.gov/home): For medical image segmentation and classification, containing 44,000 images. Split into training/validation/test sets of 32,000, 4,000, and 8,000 images, respectively.
ChestX-Det (https://github.com/Deepwise-AILab/ChestX-Det-Dataset): An expanded version of ChestX-Det10 with 3,578 images for chest-related disease detection. Split into training/validation/test sets of 2,077, 756, and 756 images, respectively.
SIXRay (https://github.com/MeioJane/SIXray): Contains over 1 million X-ray scan images of airport luggage. Includes subsets like SIXRay-100 and SIXRay-10 with varying training and test images.



This document provides an overview of the datasets used in our research, along with specific modifications and configurations.


##Μodifications and configurations.
The following modifications are contained in preprocessing.py

Places-100:
Derived from Places-365 by randomly selecting 500 images per class (total of 50,000 images), contrary to the 400 images per class in Places-365 as per \cite{he2021quantifying}.
The dataset is divided into training, validation, and test sets with a 70-15-15 split ratio, this modified variant is referred to whenever Places-100 is mentioned}.
ADE20K:
The training set is randomly split into a training and a validation set, following an 80:20 ratio.
STL-10:
The labeled images are grouped into 10 predefined folds, each comprising 50 images from a total of 500 annotated images.
One fold is used as the validation set, and the remaining 9 as the training set.
SIXRay-100:
Resized to fit a low-data regime, resulting in a subset of 300,000 images.
The dataset includes all 8,929 positive images and a random selection of negative images.
Segmented into training, validation, and test sets with a 70-15-15 split ratio\footnote{In subsequent references, this low-data variant is implied when mentioning SIXRay-100}.
SIXRay-10 and the subsampled SIXRay-100 overlap in content, but the test images from SIXRay-10 are not included in the training or validation sets of SIXRay-100.

##General Observations
The training set of the pretraining dataset is used for both SSL pretraining and supervised pretraining variants.
In single-dataset experiments:
The training set of the pretraining dataset is also used for supervised downstream finetuning, incorporating ground-truth annotation labels.
The validation set of each pretraining dataset is employed for optimal hyperparameter search.
In experiments outside the single-dataset setup:
The training set of each downstream dataset is used for supervised downstream finetuning, considering ground-truth annotation labels.
The annotated test set of each downstream dataset is used for evaluation purposes, using predictions inferred from the trained DNN.
