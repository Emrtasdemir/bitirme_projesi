"""TFLite modelinde tek görsel çıkarsama örneği (iskelet)."""
import numpy as np
import tensorflow as tf

def run_inference_tflite(tflite_path, image):
    # image: HxWx1 float32 [0,1], model girişine göre ölçekleyin
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = np.expand_dims(image, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]['index'])
    return y
