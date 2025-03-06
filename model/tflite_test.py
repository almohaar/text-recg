import tensorflow as tf
import numpy as np
import cv2
import os
from loader import load_dataset

# Load trained model
model = tf.keras.models.load_model("best_model.keras")

# Load class names
train_dir = "dataset/train"
# class_names = sorted(os.listdir(train_dir))

def predict_image(image_path):
    """Loads an image and predicts its class."""
    img = load_dataset(image_path)
    if img is None:
        return "Invalid image"
    
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    return class_names[predicted_class]

# Example usage
test_image = "test_samples/sample1.jpg"
print(f"Predicted class: {predict_image(test_image)}")
