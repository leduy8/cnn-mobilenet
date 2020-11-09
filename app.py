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

# * Disable GPU for tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mobile = tf.keras.applications.mobilenet.MobileNet()

# * Get image and pass to prepare image func then predict with MobileNet
# Image(filename='data/MobileNet-samples/1.PNG', width=300, height=200)
# preprocessed_image = prepare_image('1.PNG')
# predictions = mobile.predict(preprocessed_image)

# Image(filename='data/MobileNet-samples/2.PNG', width=300, height=200)
# preprocessed_image = prepare_image('2.PNG')
# predictions = mobile.predict(preprocessed_image)

# Image(filename='data/MobileNet-samples/3.PNG', width=300, height=200)
# preprocessed_image = prepare_image('3.PNG')
# predictions = mobile.predict(preprocessed_image)

# * Get results and get top X predictions (default is 5)
results = imagenet_utils.decode_predictions(predictions)
print(results)
