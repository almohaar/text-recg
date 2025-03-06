import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from loader import load_dataset
from cnn_model import build_model
import datetime

# Logging function
def log_info(message):
    print(f"[INFO] {message}")

# Set dataset directory
train_dir = "datasets/"

# 🔍 Function to count classes across all dataset folders
def count_classes(dataset_path):
    all_labels = []
    for parent in os.listdir(dataset_path):
        parent_path = os.path.join(dataset_path, parent)
        if os.path.isdir(parent_path):  
            labels = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
            all_labels.extend(labels)
    return len(set(all_labels)), list(set(all_labels))

num_classes, class_names = count_classes(train_dir)
log_info(f"Found {num_classes} unique classes.")
log_info(f"Class labels: {class_names}")

# 🔍 Function to remove .tif files
def remove_tif_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".tif"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                log_info(f"Removed {file_path}")

# Run cleanup before loading dataset
remove_tif_files(train_dir)

# 🏗 Load dataset
log_info("Loading dataset...")
x_train, y_train, class_names = load_dataset(train_dir)
x_val, y_val, _ = load_dataset(train_dir)
log_info(f"Loaded dataset with {len(class_names)} classes.")

# 🏷 Print class distribution
class_counts = {class_names[i]: sum(y_train[:, i]) for i in range(len(class_names))}
log_info(f"Class distribution: {class_counts}")

# Data Augmentation & Normalization
# I guress too much augmentation is affecting performance
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize images to [0,1] range
    rotation_range=0, # causing issues, formerly 20, I'm changing this to 0 for now.
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False # change to false for now.
)
train_generator = datagen.flow(x_train, y_train, batch_size=8)  # Reduce batch size, prevent memory warnings

# Ensure validation data is also rescaled
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow(x_val, y_val, batch_size=16)

# 🚀 Build model
log_info("Building model...")
model = build_model(num_classes)

# 📌 Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy")
tensorboard = TensorBoard(log_dir=f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", histogram_freq=1)

# 🔽 Reduce learning rate when validation loss stops improving
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 🏋️ Train model
log_info("Starting training...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, tensorboard, lr_scheduler]
)

# 💾 Save final model
log_info("Training complete. Saving model...")
model.save("final_model.keras")
log_info("Model saved successfully.")
