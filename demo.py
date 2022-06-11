import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras import layers
from tensorflow.keras import models

DATASET_PATH = 'data'
SOUND_LENGTH = 32000

data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[ commands != 'none']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')

    
    
model = tf.saved_model.load("model_v1_100")

sample_file = DATASET_PATH+'/unknow/017.wav'

sample_ds = preprocess_dataset(filenames)

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{commands[label[0]]}"')
  plt.show()
  
  