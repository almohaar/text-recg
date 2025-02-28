import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Image Preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Convert Image to grayscale

# Resize Images to a fixed dimension (32 x 32)
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32

EPOCHS = 50

# Check if dataset path exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

# Create plots directory if it doesn't exist
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR, exist_ok=True)

# Load Dataset(s)
def load_data(data_dir):
    images = []
    labels = []

    for category in ['capital_letters', 'small_letters']:
        category_path = os.path.join(data_dir, category)

        # each folder in the category dir
        for label in os.listdir(category_path):
            label_path = os.path.join(category_path, label)

            # each image in the label dir
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)

                try:
                    # Edit each image
                    # Turn each image into black and white
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # resize image dimension to 32 x 32 pixels
                    img = cv2.resize(img, IMAGE_SIZE)

                    # add to images array
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f'Error loading image {img_path} due to {e}')

    return np.array(images), np.array(labels)

# print(os.path.abspath(DATA_DIR)) 
# print(os.path.exists(DATA_DIR))

# Load and Preprocess dataset(s)
images, labels = load_data(DATA_DIR)

# Normalize pixel values to the range ([0, 1])
images = images / 255.0
# Add Channel Dimension
images = np.expand_dims(images, axis=-1)

# Encode labels as Integers
label_names = np.unique(labels)
label_map = { name: idx for idx, name in enumerate(label_names)}
labels = np.array([label_map[label] for label in labels])

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f'Training samples: {len(X_train)}, Testing Samples: {len(X_test)}')


# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# Compute class weights to handle data imbalance
class_weights = {i: max(len(Y_train) / (len(label_names) * np.bincount(Y_train)[i]), 1) for i in range(len(label_names))}


# Build the CNN Model with Dropout Layers
model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_names), activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# summarize
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test))

# Evaluate Model Accuracy
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=2)

# Show Test Accuracy
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot Training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
# plt.show()
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
# plt.show()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_plot.png'))
plt.close()


# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()

# Classification Report
print("\nClassification Report:\n", classification_report(Y_test, y_pred, target_names=label_names))