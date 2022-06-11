import os
import pathlib

import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import glob
import random


from IPython.display import Audio, Image
from scipy.io import wavfile

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

from tensorflow.keras import layers
from tensorflow.keras import models

DATASET_PATH = 'data'

data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[ commands != 'none']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')

    
    
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])


train_files = filenames[:110]
val_files = filenames[110: 110 + 40]
test_files = filenames[-40:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


code_to_name = {
  'pushka': 'Pushka',
  'unknow': 'Unknow'
}



