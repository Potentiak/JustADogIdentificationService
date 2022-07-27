#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:23:16 2022

@author: AR621 
"""
#%% IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import re

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# tf gpu check
print(tf.config.list_physical_devices('GPU')) 
# ^ You really want it for this one :-)
#%% PATH TO DATA
# STANDFORD DOG DATASET:
# http://vision.stanford.edu/aditya86/ImageNetDogs/

PROJECT_PATH = 'C:/Users/maxpa/Documents/Data/PESEK/'
PATH = PROJECT_PATH + 'Images/'
PATH_ANNOTATIONS = PROJECT_PATH + 'Annotations/' 
#%%% preINIT
from matplotlib.pyplot import close
# close all figures
close('all') # MATLAB vibes
#%% dataset label formatting
class_names = os.listdir(PATH)
breed_list=[]
# whoosh! Silly, non human readable class names be gone!
for breed in class_names:
    breed = re.sub("n[\d][\d][\d][\d][\d][\d][\d][\d]-", '',breed)
    breed = re.sub("_", " ", breed)
    breed_list.append(breed)
#%% dataset label identification and mapping
# save .txt with all the class names for later comiplation into tflite model
labels_file_name='classes.txt'
with open(labels_file_name, 'w', encoding='utf-8') as file:
    for i in range(len(breed_list)):
        file.write(list(breed_list)[i] + '\n')

# label_maps = {}
# label_maps_rev = {}
# for i, v in enumerate(breed_list):
#     label_maps.update({v: i})
#     label_maps_rev.update({i : v})
    
#%%% Define input batch
batch_size = 64
img_height = 224
img_width = 224
#%%% split dataset into test and train
data_split = 0.15

train_ds = tf.keras.utils.image_dataset_from_directory(
  PATH,
  validation_split=data_split,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  PATH,
  validation_split=data_split,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# a lazy way of splitting data...
# ... but a way none the less

#%% show me what you will see
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(1):
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(breed_list[labels[i]])
    plt.axis("off")
#%% normalize both validation and training data
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_val_ds))
first_image = image_batch[0]

# Finally print min and max to ensure that data sets are normalized properly
print(np.min(first_image), np.max(first_image))
print(np.min(first_image), np.max(first_image))

#%% prepare data augmentation layer
data_augmentation_layer = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(.25),
    layers.RandomContrast(.05),
    layers.RandomZoom(.15),
  ]
)

#%% show example augmented data
plt.close('all')

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation_layer(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
    
#%% CNN itself - params
num_classes = len(breed_list)

dense1=4096
dense2=dense1//2

dropout=0.5
#%% CNN itself - model(s)
# %% if one for tansfer learning...
from keras.applications.densenet import DenseNet201
# keras.Input(img_height,img_width,3)
dense_net = DenseNet201(input_shape = (img_height,img_width,3),
                        weights='imagenet',
                        include_top=False)

model = Sequential([
   data_augmentation_layer,
  
   dense_net,
  
  layers.GlobalAveragePooling2D(),
   
   layers.Dense(dense1, activation='relu'),
   layers.Dropout(dropout),
  
  layers.Dense(num_classes, activation='softmax')
  ])

# since thats the point of transfer learning
for layer in model.layers[:-4]:
    layer.trainable = False
    
model.summary()

# %% ...or one for classicall cnn

# model = Sequential([
#    data_augmentation_layer,
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(conv1_neurons, 3, padding='same', activation='relu'),
#     # layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.AveragePooling2D(),
#   layers.Conv2D(conv2_neurons, 3, padding='same', activation='relu'),
#   layers.AveragePooling2D(),
#   layers.Conv2D(conv3_neurons, 3, padding='same', activation='relu'),
#   layers.AveragePooling2D(),
#   layers.Flatten(),
  
#    # layers.Dense(1024, activation='relu'),
#   layers.Dense(1920, activation='relu'),
#   layers.Dropout(0.5),
#   # layers.Dense(960, activation='relu'),
#   # layers.Dropout(0.5),
#   layers.Dense(num_classes)
# ])

# model.summary()

# %% compile the model (and summerize)
opt = 'adamax'
# opt = 'adamax'
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#%% visualization
import visualkeras
visualkeras.layered_view(model).show()
#%% everyday hard training
epochs=25
history = model.fit(
  normalized_ds,
  validation_data=normalized_val_ds,
   epochs=epochs
)
#%% show me how you see

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.suptitle('Training history for model using transfer learning')

ax1 = plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('number of epochs')
plt.ylabel('classification accurracy')
plt.ylim(0,1) # so that it is nice and comparable to other charts diaplying accurracy
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('number of epochs')
plt.ylabel('classification accurracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
# %% save the model...
tf.saved_model.save(model, PROJECT_PATH)
#%% ...or load if you already have one
model = keras.models.load_model(PROJECT_PATH)
# %% Convert the model. (to TFLite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

save_dir = 'C:/Users/maxpa/PycharmProjects/NN/CLASSIFICATION/TFLites/'
save_name = 'lupus-omni-die.tflite'
# Save the lite model.
with open(save_dir + save_name, 'wb') as f:
  f.write(tflite_model)
  
# %% create metadata for the tflite model
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils

ImageClassifierWriter = image_classifier.MetadataWriter
_MODEL_PATH = save_dir + save_name
# Task Library expects label files that are in the same format as the one below.
_LABEL_FILE = labels_file_name

_SAVE_TO_PATH = os.curdir
# Normalization parameters is required when reprocessing the image. It is
# optional if the image pixel values are in range of [0, 255] and the input
# tensor is quantized to uint8. See the introduction for normalization and
# quantization parameters below for more details.
# https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters)
_INPUT_NORM_MEAN = 0
_INPUT_NORM_STD = 255

# Create the metadata writer.
writer = ImageClassifierWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
    [_LABEL_FILE])

# Verify the metadata generated by metadata writer.
print(writer.get_metadata_json())

# Populate the metadata into the model.
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

#%% Tests
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    for i in range(1):
      img = images[i].numpy().astype("uint8")
      plt.imshow(img)
      prediction = breed_list[predictions[i].argmax()]
      
      plt.title( "True label: %s, prediction: %s" %(breed_list[labels[i]], prediction))
      plt.axis("off")
      
#%% show specific image
import cv2
model_classify = model
# TEST_PATH = DATA_PATH='C:/Users/maxpa/Data/MASKNET/test/'
# TEST_PATH = "/media/Killshot/KSPOCKET/PESEK/test/"
TEST_PATH = "G:/PESEK/test/"
IMAGE_NAME = "elkhund"
# IMAGE_NAME = "lab1"
# IMAGE_NAME = "shit"
# IMAGE_NAME = "muzzle"
# IMAGE_NAME = "muzzle1"

# IMAGE_NAME = IMAGE_NAME + ".png"
IMAGE_NAME = IMAGE_NAME + ".jpg"

img = cv2.imread(TEST_PATH + IMAGE_NAME)

# prepare image to be same shape as training data
final_img=cv2.resize(img, [160, 160]) 
final_img = final_img/255
final_img=np.reshape(final_img, [1,final_img.shape[0], final_img.shape[1], final_img.shape[2]])


# convert image to rgb (for display purposes only)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# add rectangles around detected faces
mask_classification=model_classify.predict(final_img)
percentage = mask_classification.max()*100
# percentage=percentage[0,-1]*100
percentage=round(percentage,1)
print('\n-> ' + str(breed_list[mask_classification.argmax()]) + ' : ' + str(percentage) +  '%')

# show input image
plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(img)
plt.title(':%s:::%s%%:' %(str(breed_list[mask_classification.argmax()]), percentage))