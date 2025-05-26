import numpy as np
import tensorflow as tf
import ai_edge_litert.interpreter as tflite


import os
from sys import argv

from detector_config import config as cfg

config = cfg.Config()

DEFAULT_DATATYPE = "test"


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


def load_dataset_with_paths(data_dir, image_size=(224, 224), batch_size=16):
    class_names = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())

    all_file_paths = []
    all_labels = []
    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_file_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label_index)
    
    path_ds = tf.data.Dataset.from_tensor_slices((all_file_paths, all_labels))

    def process_dataset_path(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.uint8)
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

def process_image_path(image_path):
    image_size = (224, 224)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.uint8)
    image = tf.expand_dims(image, 0).numpy()
    return image

# Combines some of the logic from grabbing images from a 
# directory into a dataset (namely processing a path into
# an image object) as well as the logic of performing 
# inferencing on a single image
def inference_image(image_path, interpreter=None):
    class_names = ["no-squirrel", "squirrel"]
    interpreter = get_interpreter() if interpreter is None else interpreter
    image = process_image_path(image_path)
    output = run_inference_tflite(interpreter, image)
    predicted = class_names[np.argmax(output)]
    print(f"Inferencing result: {image_path} is a {predicted}")
    return predicted

def inference_data_dir(datatype="test"):
    all_predictions = []
    all_labels = []
    misclassified = []

    data_dir = f"{config.data_dir}/{datatype}"

    dataset = load_dataset(data_dir, batch_size=40)
    class_names = dataset.class_names

    interpreter = get_interpreter()
    for batch in dataset:
        images, labels, paths = batch    
        for i in range(images.shape[0]):
            image = tf.expand_dims(images[i], 0).numpy()

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

if __name__ == "__main__":
    inference_type = argv[1]
    inference_location = argv[2] if len(argv) > 2 else None
    if inference_type == "directory":
        inference_location = DEFAULT_DATATYPE if inference_location is None else inference_location
        inference_data_dir(inference_location)
    else:
        inference_image(inference_location)