import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random

def load_dataset(dataset_path, dataset_name):
    """
    Load images from a given dataset path.
    Returns a list of image arrays and their labels.
    Custom handling for different datasets.
    """
    images = []
    labels = []

    if dataset_name in ['Places-100', 'SIXRay-100']:
        # Handling for Places-100 and SIXRay-100
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            image_files = os.listdir(folder_path)
            # Randomly select a subset of images if required
            if dataset_name == 'Places-100':
                image_files = random.sample(image_files, 500)
            for image_name in image_files:
                process_image(folder_path, image_name, images, labels, folder_name)

    elif dataset_name == 'ADE20K':
        # Handling for ADE20K
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            for image_name in os.listdir(folder_path):
                process_image(folder_path, image_name, images, labels, folder_name)

    elif dataset_name == 'STL-10':
        # Handling for STL-10
        for fold in range(10):
            fold_path = os.path.join(dataset_path, f"fold_{fold}")
            for image_name in os.listdir(fold_path):
                process_image(fold_path, image_name, images, labels, fold)

    return images, labels

def process_image(folder_path, image_name, images, labels, label):
    """
    Helper function to process and add an image to the dataset.
    """
    image_path = os.path.join(folder_path, image_name)
    try:
        image = Image.open(image_path)
        images.append(np.array(image))
        labels.append(label)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")

# ... [Rest of the functions like 'preprocess_images' and 'split_dataset' remain the same]

# Example usage for Places-100
places100_path = "path/to/places100"
images, labels = load_dataset(places100_path, 'Places-100')
images = preprocess_images(images)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)

# You can repeat the above example usage for ADE20K, STL-10, and SIXRay-100 by changing the path and dataset name.
