import scipy.io as cio
import numpy as np
import keras
from keras.layers.core import Layer
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten

import cv2
from keras import backend as K
from keras.utils import np_utils

import math
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import os
import matplotlib.image as mpg
from keras.models import Sequential



IMG_SIZE = 224
loadmodel_names = cio.loadmat("D:/compCarsThesisData/data/misc/make_model_name.mat")
model_names = loadmodel_names['model_names']

model_names_dict = []
count = 0
for i in range(model_names.size):
  if ((len(model_names[i][0])) >= 1):
    model_names_dict.append(model_names[i][0][0])
  elif ((len(model_names[i][0])) < 1):
    model_names_dict.append('Null')

# print("MAKE_NAMES AS A STRING: ", model_names_dict)
labels_dict = {v: k for v, k in enumerate(model_names_dict)}
print(labels_dict)

train_count = 0
test_count = 0


train = open("D:/compCarsThesisData/data/train_test_split/classification/train -10000.txt","r")

path = "D:/compCarsThesisData/data/image/"

training_images = []
training_labels = []

for x in train:
# ===================================================
# Concating the path with train.txt path line by line and reading the images for training
# ===================================================
  abspath = str(path+x).rstrip("\n\r")

# ===================================================
# Matplotlib.image reading images from path function 
# =================================================== 
  train_image_reader = mpg.imread(abspath)

  train_image_reader_preprocess = train_image_reader / 255.0
# ================================= 
# Resizing Of Image By Using Opencv
# =================================
  train_image = cv2.cv2.resize(train_image_reader_preprocess,(IMG_SIZE,IMG_SIZE))
  # print("TrainIMAge",train_image)
  # train_image_preprocess = train_image / 255.0
  # print("Preprocessed",train_image_preprocess)
# =================================
# Getting Ready with Training Data
# =================================
  training_images.append(np.asarray(train_image))

  # plt.imshow(training_images[train_count])
  # plt.show()
# ================================================
# Array for 163 make_names for Labeling of Images
# ================================================
  arr = np.zeros(2004)

# ====================================  
# Creating the hot encoded labels 
# ====================================
  arr[int(abspath.rsplit('/')[5])-1] = 1
  
# ====================================
# Getting Ready With Training Labels
# ====================================
  training_labels.append(np.asarray(arr))
  train_count=train_count + 1
  print(train_count, arr, abspath)
  # exit()
  # exit()
  import gc
  gc.collect()
print("TRAINING AND LABELS IMAGES DONE")


test = open("D:/compCarsThesisData/data/train_test_split/classification/test - 8000.txt","r")

print("TEST DATA LOADING....")

testing_images = []
testing_labels = []
for y in test:

# ===================================================
# Concating the path with test.txt path line by line and reading the images for testing
# ===================================================
  abspath = str(path+y).rstrip("\n\r")
  # print(abspath)

# ===================================================
# Matplotlib.image reading images from path function 
# =================================================== 
  test_image_reader = mpg.imread(abspath)
  test_image_reader_preprocess = test_image_reader / 255.0
# ================================= 
# Resizing Of Image By Using Opencv
# =================================
  test_image = cv2.cv2.resize(test_image_reader_preprocess,(IMG_SIZE,IMG_SIZE))
 

# =================================
# Getting Ready with Testing Data
# =================================
  # print(image_reader.shape)
  testing_images.append(np.asarray(test_image))


# ================================================
# Array for 163 make_names for Labeling of Images
# ================================================
  arr = np.zeros(2004)
  # print(abspath.rsplit('/')[4])

# ====================================  
# Creating the hot encoded labels 
# ====================================
  arr[int(abspath.rsplit('/')[5])-1] = 1
  # print("Array Label",arr)


# ====================================
# Getting Ready With Testing Labels
# ====================================
  testing_labels.append((arr))
  test_count = test_count + 1
  print(test_count, arr, abspath)
  gc.collect()

print("TEST AND LABELS ARE DONE NOW...")


np.save('2004_labels_Inception_224_Training_data',training_images)
np.save('2004_labels_Inception_224_Training_labels', training_labels)
np.save('2004_labels_Inception_224_Testing_data', testing_images)
np.save('2004_labels_Inception_224_Testing_labels',testing_labels)