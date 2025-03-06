import tensorflow as tf
import os

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../tflite/yoruba_model.h5'))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
tflite_path = os.path.join(os.path.dirname(__file__), '../tflite/model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Converted TensorFlow Lite model saved at {tflite_path}")
