from prepare_image import prepare_image

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from IPython.display import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
y = tf.keras.layers.Dense(4, activation=tf.nn.relu)(x)
z = tf.keras.layers.Dense(4, activation=tf.nn.relu)(y)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(z)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

x2 = model.layers[-3].output
outputs2 = Dense(units=10, activation='softmax')(x2)
model2 = Model(inputs=model.input, outputs=outputs2)

model2.summary()
