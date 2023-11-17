#Datasets Summary



This document provides an overview of the datasets used in our research, along with specific modifications and configurations.


##Μodifications and configurations.

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
