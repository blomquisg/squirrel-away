import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import datetime
import os
import tempfile

from trainer_config import config as cfg
import trainer.dataset as ds

config = cfg.Config()

num_classes = 2  # squirrel or no_squirrel
num_epochs = 15
img_size = (224, 224)
batch_size = 32

train_data_dir = f"{config.data_dir}/{ds.DatasetType.Train.value}"
data_generator = ImageDataGenerator(rescale=1./255)

training_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical" # for softmax
)

# get the base model we will extend and freeze it
base_model = MobileNetV2(weights="imagenet", 
                         include_top=False, 
                         input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

training_results = ""
for epoch in range(num_epochs):
    history = model.fit(training_generator, epochs=1, verbose=1)
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    epoch_results = f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
    training_results += epoch_results + "\n"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
trained_dir = f"{config.experiments_dir}/trained"
experiment_dir = f"{trained_dir}/{timestamp}"
os.makedirs(experiment_dir, exist_ok=True)

model_filename = f"{experiment_dir}/model.keras"
training_results_filename = f"{experiment_dir}/training_results.txt"

model.save(model_filename)
config.logger.debug(f"Model saved to {model_filename}")

def _update_symlink(target, link):
    temp_link = tempfile.mktemp(prefix=link + "_tmp")
    os.symlink(target, temp_link)
    os.replace(temp_link, link)

_update_symlink(experiment_dir, f"{trained_dir}/latest")

with open(training_results_filename, "w") as results_file:
    results_file.write(training_results)

config.logger.debug(f"Training results saved to {training_results_filename}")


# Quantize the model and convert to tflite (for Coral TPU)

def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]
    
model = tf.keras.models.load_model(model_filename)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(f"{experiment_dir}/model.tflite", "wb") as tflite_file:
    tflite_file.write(tflite_model)