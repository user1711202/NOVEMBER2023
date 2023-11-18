import os
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_sixray100_dataset(dataset_path):
    """
    Load and preprocess SIXRay-100 dataset.
    Returns images and labels, with the dataset split according to the specified ratio.
    """
    images, labels = load_dataset_generic(dataset_path, sample_per_class=None)
    return split_dataset(images, labels, test_size=0.15, val_size=0.15)

def load_places100_dataset(dataset_path):
    """
    Load and preprocess Places-100 dataset.
    Returns images and labels, with the dataset split according to the specified ratio.
    """
    images, labels = load_dataset_generic(dataset_path, sample_per_class=500)
    return split_dataset(images, labels, test_size=0.15, val_size=0.15)

def load_dataset_generic(dataset_path, sample_per_class=None):
    """
    Generic dataset loading function.
    """
    images = []
    labels = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        image_files = os.listdir(folder_path)
        if sample_per_class:
            image_files = random.sample(image_files, min(len(image_files), sample_per_class))
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            try:
                image = Image.open(image_path)
                images.append(np.array(image))
                labels.append(folder_name)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return images, labels

def preprocess_images(images, target_size=(224, 224)):
    """
    Resize and normalize images.
    """
    processed_images = [Image.fromarray(img).resize(target_size) for img in images]
    processed_images = np.array(processed_images) / 255.0
    return processed_images

def split_dataset(images, labels, test_size=0.15, val_size=0.15):
    """
    Split the dataset into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size based on remaining dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage for SIXRay-100
sixray100_path = "path/to/sixray100"
images, labels = load_sixray100_dataset(sixray100_path)
images = preprocess_images(images)
X_train_sixray, X_val_sixray, X_test_sixray, y_train_sixray, y_val_sixray, y_test_sixray = split_dataset(images, labels)

# Example usage for Places-100
places100_path = "path/to/places100"
images, labels = load_places100_dataset(places100_path)
images = preprocess_images(images)
X_train_places, X_val_places, X_test_places, y_train_places, y_val_places, y_test_places = split_dataset(images, labels)
