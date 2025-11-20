import sys
print("Python:", sys.version.split()[0])

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("TensorFlow import error:", e)

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV import error:", e)

try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy import error:", e)
