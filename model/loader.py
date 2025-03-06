import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = (32, 32)  # Define image size

def load_dataset(data_dir, categories=None):
    """Loads dataset from multiple category folders, processes images, and encodes labels."""
    images, labels = [], []
    class_names = set()  # To collect unique class names

    # Default dataset categories if none are provided
    if categories is None:
        categories = ['capital_letters', 'small_letters', 'dataset_1', 'dataset_2', 'upper']

    for category in categories:
        category_path = os.path.join(data_dir, category)

        if not os.path.isdir(category_path):
            print(f"⚠️ Skipping missing category: {category_path}")
            continue

        for label in sorted(os.listdir(category_path)):  # Sort labels for consistency
            label_path = os.path.join(category_path, label)

            if not os.path.isdir(label_path):
                print(f"⚠️ Skipping non-directory: {label_path}")
                continue

            class_names.add(label)

            for img_file in sorted(os.listdir(label_path)):  # Sort files for consistency
                img_path = os.path.join(label_path, img_file)

                # Skip non-image files
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"⚠️ Skipping unreadable image: {img_path}")
                        continue

                    img = cv2.resize(img, IMAGE_SIZE)
                    img = img.astype('float32') / 255.0  # Normalize

                    images.append(img)
                    labels.append(label)

                except Exception as e:
                    print(f"⚠️ Skipping corrupt image: {img_path}, Error: {e}")

    # Ensure dataset is not empty
    if len(images) == 0 or len(labels) == 0:
        raise ValueError(f"No images were loaded from {data_dir}. Please check the dataset structure!")

    # Sort class names for consistent mapping
    class_names = sorted(class_names)
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([label_map[label] for label in labels])

    labels = to_categorical(labels, num_classes=len(class_names))  # One-hot encoding
    images = np.array(images).reshape(-1, 32, 32, 1)  # Reshape for CNN input

    print(f"Loaded {len(images)} images from {len(class_names)} classes.")
    return images, labels, class_names
