import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


DATASET_PATH = 'data'
SOUND_LENGTH = 32000

data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[ commands != 'none']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')

    
for filename in filenames:
    test_file = tf.io.read_file(filename)
    test_audio, sample_rate = tf.audio.decode_wav(contents=test_file)
    print(filename)
    print(test_audio.shape)
    if test_audio.shape[0]<SOUND_LENGTH:
        t0 = tf.tile([[0.0]],[SOUND_LENGTH-test_audio.shape[0],1])
        test_audio = tf.concat([test_audio, t0],0)
        test_audio = tf.reshape(test_audio, [SOUND_LENGTH,1])
        test_file = tf.audio.encode_wav(test_audio, sample_rate)
        tf.io.write_file(filename, contents=test_file)
        print(filename,test_audio.shape)

