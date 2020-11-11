from prepare_image import prepare_image
from plot_confusion_matrix import plot_confusion_matrix

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from IPython.display import Image

from pathlib import Path
Path("./models").mkdir(parents=True, exist_ok=True)

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
# results = imagenet_utils.decode_predictions(predictions)
# print(results)

# * Distributed images to train, valid and test folder
# os.chdir('data/Sign-Language-Digits-Dataset')
# if os.path.isdir('train/0/') is False:
#     os.mkdir('train')
#     os.mkdir('valid')
#     os.mkdir('test')

#     for i in range(0, 10):
#         shutil.move(f'{i}', 'train')
#         os.mkdir(f'valid/{i}')
#         os.mkdir(f'test/{i}')

#         valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
#         for j in valid_samples:
#             shutil.move(f'train/{i}/{j}', f'valid/{i}')

#         test_samples = random.sample(os.listdir(f'train/{i}'), 10)
#         for k in test_samples:
#             shutil.move(f'train/{i}/{k}', f'test/{i}')
# os.chdir('../../')

# * Process the data
train_path = 'data/Sign-Language-Digits-Dataset/train'
valid_path = 'data/Sign-Language-Digits-Dataset/valid'
test_path = 'data/Sign-Language-Digits-Dataset/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=10)
# ? In test batches, we set shuffle to False, so that we can access the corresponding non-shuffle test labels to plot in the confustion matrix
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

# * Grab layers from start at the last sixth layer in mobilenet model (i.e grab all except last 5 layers in mobilenet)
# x = mobile.layers[-6].output

# * Create an output layer
# ? This is Keras Functional API syntax, this is chain layer calls to specify the model's forward pass
# output = Dense(units=10, activation='softmax')(x)

# * Create the fine-tuned model
# model = Model(inputs=mobile.input, outputs=output)

# * Freeze the weight, only train last 23 layer for new dataset
# ? 23 is not a magic number, it's just an experiment and find it work pretty decently
# for layer in model.layers[:-23]:
#     layer.trainable = False

# # model.summary()

# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x=train_batches, steps_per_epoch=len(train_batches),
#           validation_data=valid_batches, validation_steps=len(valid_batches), epochs=30, verbose=2)

# * Save model(the architecture, the weights, the optimizer, the state of the optimizer, the learning rate, the loss, etc.) to a .h5 file
# ? If found a model, delete it and save a new one
# if os.path.isfile("models/fine-tune_mobilenet_model.h5") is True:
#     os.remove("models/fine-tune_mobilenet_model.h5")
# model.save(r"models/fine-tune_mobilenet_model.h5")

# * Load model
model = load_model(r'models/fine-tune_mobilenet_model.h5')
# * Freeze the weight, only train last 23 layer for new dataset
# ? 23 is not a magic number, it's just an experiment and find it work pretty decently
for layer in model.layers[:-23]:
    layer.trainable = False

# * Get unshuffle test labels
test_labels = test_batches.classes

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
