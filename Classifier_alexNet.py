import os 
import matplotlib.image as mpg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy.io as cio
from sklearn.externals import joblib
from tflearn import models
from tflearn.layers.normalization import local_response_normalization
# AUTHOR: FARRUKH AHMED KHAN
# EMAIL : farukhcs15@gmail.com

IMG_SIZE = 64
MODEL_NAME = 'Vehicle Classification'


# LR = 1e-3
train_count = 0
test_count = 0

# # =======================================
# # Reading Mat File With Scipy.io
# # =======================================
make_names = cio.loadmat("D:/compCarsThesisData/data/misc/make_model_name.mat")

# # ===========================================
# # Saving cell array make_names from the file
# # ===========================================
make_names_all = make_names['make_names']


# # =======================================
# # Building the dictionary
# # =======================================
make_names_dict = []
count = 0
for i in range(make_names_all.size):
  make_names_dict.append(make_names_all[i][0][0])

print("MAKE_NAMES AS A STRING: ", make_names_dict)
labels_dict = {v: k for v, k in enumerate(make_names_dict)}

print("MAKE_NAMES_LABEL_DICTIONARY: ",labels_dict)
# =======================================
# Path For Training Split From Research
# =======================================
train = open("D:/compCarsThesisData/data/train_test_split/classification/train.txt","r")

# ==========================
# Path For the Image Data
# ==========================
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

# ================================= 
# Resizing Of Image By Using Opencv
# =================================
  train_image = cv2.cv2.resize(train_image_reader,(IMG_SIZE,IMG_SIZE))
 
# =================================
# Getting Ready with Training Data
# =================================
  training_images.append(np.asarray(train_image))

  # plt.imshow(training_images[train_count])
  # plt.show()
# ================================================
# Array for 163 make_names for Labeling of Images
# ================================================
  arr = np.zeros(163)

# ====================================  
# Creating the hot encoded labels 
# ====================================
  arr[int(abspath.rsplit('/')[4])-1] = 1

# ====================================
# Getting Ready With Training Labels
# ====================================
  training_labels.append(np.asarray(arr))
  train_count=train_count + 1
  print(train_count)

print("TRAINING AND LABELS IMAGES DONE")



test = open("D:/compCarsThesisData/data/train_test_split/classification/test.txt","r")

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

# ================================= 
# Resizing Of Image By Using Opencv
# =================================
  test_image = cv2.cv2.resize(test_image_reader,(IMG_SIZE,IMG_SIZE))


# =================================
# Getting Ready with Testing Data
# =================================
  # print(image_reader.shape)
  testing_images.append((test_image))


# ================================================
# Array for 163 make_names for Labeling of Images
# ================================================
  arr = np.zeros(163)
  # print(abspath.rsplit('/')[4])

# ====================================  
# Creating the hot encoded labels 
# ====================================
  arr[int(abspath.rsplit('/')[4])-1] = 1
  # print("Array Label",arr)


# ====================================
# Getting Ready With Testing Labels
# ====================================
  testing_labels.append((arr))
  test_count = test_count + 1
  print(test_count)

print("TEST AND LABELS ARE DONE NOW...")
  # print("0th Index Appended Label", testing_labels[0])

# For Saving the data in npy format
np.save('Training_data_alexNet',training_images)
np.save('Training_labels_alexNet', training_labels)
np.save('Testing_data_alexNet', testing_images)
np.save('Testing_labels_alexNet',testing_labels)

# =========================
# Data already Saved, Load it
# =========================
# X_train = np.load('D:/ThesisWork/Training_data.npy')#training_images
# y_train = np.load('D:/ThesisWork/Training_labels.npy')#training_labels
# X_test = np.load('D:/ThesisWork/Testing_data.npy')#testing_images
# y_test = np.load('D:/ThesisWork/Testing_labels.npy')#testing_labels

X_train = training_images
y_train = training_labels
X_test = testing_images
y_test = testing_labels

with tf.device('/gpu:0'):
  tf.reset_default_graph()
  network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
  network = conv_2d(network, 96, 11, strides=4, activation='relu')
  network = max_pool_2d(network, 3, strides=2)
  network = local_response_normalization(network)
  network = conv_2d(network, 256, 5, activation='relu')
  network = max_pool_2d(network, 3, strides=2)
  network = local_response_normalization(network)
  network = conv_2d(network, 384, 3, activation='relu')
  network = conv_2d(network, 384, 3, activation='relu')
  network = conv_2d(network, 256, 3, activation='relu')
  network = max_pool_2d(network, 3, strides=2)
  network = local_response_normalization(network)
  network = fully_connected(network, 4096, activation='tanh')
  network = dropout(network, 0.5)
  network = fully_connected(network, 4096, activation='tanh')
  network = dropout(network, 0.5)
  network = fully_connected(network, 163, activation='softmax')
  network = regression(network, optimizer='momentum',
                      loss='categorical_crossentropy',
                      learning_rate=0.001, name = 'targets')
  model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
  history = model.fit({'input': X_train}, {'targets': y_train}, n_epoch=40,
            validation_set=({'input': X_test}, {'targets': y_test}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)