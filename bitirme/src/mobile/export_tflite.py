"""Keras modelini TFLite formatına export eder."""
import tensorflow as tf

def export_to_tflite(keras_model_path, tflite_out_path, quantize_dynamic=True):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out_path, "wb") as f:
        f.write(tflite_model)
    return tflite_out_path
