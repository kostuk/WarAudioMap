import os
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))

import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
import tflite_model_maker as mm
print(f"Model Maker Version: {mm.__version__}")