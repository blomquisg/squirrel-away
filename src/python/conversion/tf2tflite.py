import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("experiments/trained_20250302_183521/squirrel_model.tf")
tflite_model = converter.convert()
with open("experiments/trained_20250302_183521/squirrel_model.tflite", "wb") as f:
    f.write(tflite_model)

