import tensorflow as tf
import keras
from keras.layers.convolutional import Conv1D, MaxPooling1D

import numpy as np

x = np.zeros((1, 249, 4))
l = keras.layers.Conv1D(256, 7, padding='same')(x)
