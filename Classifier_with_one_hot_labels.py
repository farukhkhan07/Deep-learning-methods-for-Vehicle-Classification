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

IMG_SIZE = 200
MODEL_NAME = 'Vehicle Classification'
LR = 1e-3
train_count = 0
test_count = 0
train = open("D:/compCarsThesisData/data/train_test_split/classification/train.txt","r")

path = "D:/compCarsThesisData/data/image/"
training_images = []
training_labels = []
for x in train:
  abspath = str(path+x).rstrip("\n\r")
  # print(abspath)
  train_image_reader = mpg.imread(abspath)
  train_image = cv2.cv2.resize(train_image_reader,(IMG_SIZE,IMG_SIZE))

  # print(image_reader.shape)
  training_images.append(np.asarray(train_image))

  arr = np.zeros(163)
  # print(abspath.rsplit('/')[4])
  arr[int(abspath.rsplit('/')[4])-1] = 1
  # print("Array Label",arr)
  training_labels.append(arr)
  train_count=train_count + 1
  print(train_count)

print("TRAINING AND LABELS IMAGES DONE")

  # print("0th Index Appended Label", training_labels[0])

test = open("D:/compCarsThesisData/data/train_test_split/classification/test.txt","r")
print("TEST DATA LOADING....")
testing_images = []
testing_labels = []
for y in test:
  abspath = str(path+y).rstrip("\n\r")
  # print(abspath)
  test_image_reader = mpg.imread(abspath)
  test_image = cv2.cv2.resize(test_image_reader,(IMG_SIZE,IMG_SIZE))

  # print(image_reader.shape)
  testing_images.append(np.asarray(test_image))

  arr = np.zeros(163)
  # print(abspath.rsplit('/')[4])
  arr[int(abspath.rsplit('/')[4])-1] = 1
  # print("Array Label",arr)
  testing_labels.append(arr)
  test_count = test_count + 1
  print(test_count)

print("TEST AND LABELS ARE DONE NOW...")
  # print("0th Index Appended Label", testing_labels[0])
  # exit()

X_train = training_images
y_train = training_labels
X_test = testing_images
y_test = testing_labels

tf.reset_default_graph()
convnet = input_data(shape=(None,IMG_SIZE,IMG_SIZE,3),name='input')
#shape=[None, IMG_SIZE, IMG_SIZE, 1],
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 163, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
