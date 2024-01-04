import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATASET_PATH = 'data/pushka'
data_dir = pathlib.Path(DATASET_PATH)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
print(filenames)