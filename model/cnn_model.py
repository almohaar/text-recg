import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_model(num_classes):
    """Creates a CNN model for Yoruba character recognition."""
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Reduce overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Match dataset classes
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005), # I'm trying 0.0005, 0.001 is too high
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
