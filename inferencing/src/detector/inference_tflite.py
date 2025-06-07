import numpy as np
import tensorflow as tf
import ai_edge_litert.interpreter as tflite

import os
import sys
from sys import argv
from pathlib import Path
import shutil

from detector_config import config as cfg

import time
global_start = time.perf_counter()

DEFAULT_DATATYPE = "test"

image_match_globs = ("*.jpg", "*.jpeg", "*.png")

class timing:
    def __init__(self, label=None, global_start=None):
        self.label = label
        self.global_start = global_start or time.perf_counter()

    def __enter__(self):
        self.start = time.perf_counter()
        return self # allow access to `start` or `duration`

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        print(f"Time[{self.label or 'block'}]: {self.end - self.start:.6f} ({self.end - self.global_start:.6f})")

with timing("create config", global_start):
    config = cfg.Config()

class DatasetWithPaths:
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def map(self, *args, **kwargs):
        return self.dataset.map(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.dataset.batch(*args, **kwargs)

    def unbatch(self):
        return self.dataset.unbatch()

    def take(self, count):
        return self.dataset.take(count)


def process_image_path(image_path):
    image_size = (224, 224)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.uint8)
    image = tf.expand_dims(image, 0).numpy()
    return image

def find_images_in_directory(dir):
    path = Path(dir)
    image_files = []
    for glob in image_match_globs:
        files = list(path.glob(glob))
        image_files.extend(map(str, files))
    return image_files

def load_dataset_with_paths(data_dir, image_size=(224, 224), batch_size=16):
    class_names = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())

    all_file_paths = []
    all_labels = []
    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = find_images_in_directory(class_dir)
        for image_file in image_files:
            all_file_paths.append(os.path.join(class_dir, image_file))
            all_labels.append(label_index)

    path_ds = tf.data.Dataset.from_tensor_slices((all_file_paths, all_labels))

    def process_dataset_path(file_path, label):
        image = process_image_path(image_path)
        return image, label, file_path

    dataset = path_ds.map(process_dataset_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return DatasetWithPaths(dataset, class_names)


def load_dataset(data_dir, image_size=(224, 224), batch_size=16):
#    return tf.keras.preprocessing.image_dataset_from_directory(
#        data_dir,
#        image_size=image_size,
#        batch_size=batch_size,
#        labels="inferred",
#        shuffle=True
#    )

    return load_dataset_with_paths(data_dir, image_size, batch_size)

def run_inference_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_dtype = input_details[0]["dtype"]
    assert image.dtype == expected_dtype, f"Input dtype mismatch: got {image.dtype}, expected {expected_dtype}"

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def get_interpreter():
    delegate = tflite.load_delegate(config.libedgetpu_so)

    interpreter = tflite.Interpreter(
            model_path=f"{config.model_dir}/model.tflite",
            experimental_delegates=[delegate])

    interpreter.allocate_tensors()
    return interpreter


# Combines some of the logic from grabbing images from a
# directory into a dataset (namely processing a path into
# an image object) as well as the logic of performing
# inferencing on a single image
def inference_image(image_path, interpreter=None):
    class_names = ["no-squirrel", "squirrel"]
    if interpreter is None:
        with timing("get_interpreter", global_start):
            interpreter = get_interpreter()

    if interpreter == None:
        print("What the heck!??")
    with timing("process_image_path", global_start):
        image = process_image_path(image_path)

    with timing("run_inference_tflite", global_start):
        output = run_inference_tflite(interpreter, image)

    predicted = class_names[np.argmax(output)]
    print(f"Inferencing result: {image_path} is a {predicted}")
    return predicted


def inference_directory_without_labels(dir):
    """
    """
    with timing("get_interpreter", global_start):
        interpreter = get_interpreter()
    for image_file in find_images_in_directory(dir):
        predicted = inference_image(image_file, interpreter)
        yield image_file, predicted

def inference_directory_with_labels(dir):
    """
    Performs inferencing on images in a directory structure where `dir`s
    subdirectories' names are the labels used by the squirrel detection model.

    Specifically, there should be "squirrel" and "non-squirrel" subdirectories,
    each containing any number of images.  The images should agree with the
    directory names.  E.g., the "squirrel" subdirectory should contain images
    with squirrels present.

    This method outputs results to stdout.

    Args:
        dir: The directory to find labeled subdirectories containing images for
        inferencing

    Returns:
        None
        Change this to return an "accuracy report" dict containing
          - Predictions vs. Actuals
          - Misclassified images
          - Validation accuracy totals
    """
    all_predictions = []
    all_labels = []
    misclassified = []

    dataset = load_dataset(dir, batch_size=40)
    class_names = dataset.class_names

    interpreter = get_interpreter()
    for batch in dataset:
        images, labels, paths = batch
        for i in range(images.shape[0]):
            #image = tf.expand_dims(images[i], 0).numpy()

            output = run_inference_tflite(interpreter, image)
            predicted = np.argmax(output)
            label = labels[i].numpy()

            out_string = f"Predicted: {class_names[predicted]}, Actual: {class_names[label]}"

            all_predictions.append(predicted)
            all_labels.append(label)

            if predicted != label:
                misclassified.append((f"Image_{len(all_predictions)}", class_names[label], class_names[predicted]))
                out_string = f"{out_string}\t***WRONG***"
                out_string = f"{out_string}\n\t{paths[i]}"

            print(out_string)

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"Validation accuracy: {accuracy:.4f}")

def inference_validation():
    """
    Performs inference validation on images in the Data Directory (data_dir).
    The data_dir contains validation images that can be used to check the
    accuracy of the squirrel recognition model.

    The data_dir is configured via the detector_config.

    This method outputs the inferencing results to stdout.

    Returns:
        None
    """
    dir = f"{config.data_dir}/validation"
    inference_directory_with_labels(dir)

def inference_test():
    """
    Performs inference testing on images in the Data Directory (data_dir).
    The data_dir contains test images that can be used to check the accuracy
    of the squirrel recognition model.

    The data_dir is configured via the detector_config.

    This method outputs the inferencing results to stdout.

    Returns:
        None
    """
    dir = f"{config.data_dir}/test"
    inference_directory_with_labels(dir)


def move_image(image_path, new_directory):
    image_name = os.path.basename(image_path)
    target_path = os.path.join(new_directory, image_name)
    shutil.move(image_path, target_path)


# Entry point for "squirrel-away-detector" script as defined in the pyproject.toml
# argv[1]: directory containing images for inferencing ... config.inference_dir/input
def main():
    images_path = argv[1]
    for image_file, prediction in inference_directory_without_labels(images_path):
        if prediction == "no-squirrel":
            # trash images without a squirrel
            move_image(image_file, config.no_squirrel_dir)
        else:
            # keep images with squirrels
            move_image(image_file, config.squirrel_dir)

if __name__ == "__main__":
    # TODO: Change argv so that len(argv)==1 means that it is a single image path. And,
    # len(argv)>1 means that this is a test run against a "test" or "validation" dataset
    inference_type = argv[1]
    inference_location = argv[2] if len(argv) > 2 else None
    if inference_type == "directory":
        inference_location = DEFAULT_DATATYPE if inference_location is None else inference_location
        inference_data_dir(inference_location)
    else:
        main()
