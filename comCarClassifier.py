import numpy as np 
import os
from keras.models import Sequential
import scipy.io as cio
import matplotlib.image as mpg
import matplotlib.pyplot as plt
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import cv2
from tqdm import tqdm
# data = "D:/compCarsThesisData/data/"
image = "D:/compCarsThesisData/data/image/"
# label = "D:/compCarsThesisData/data/label"
# attributes = "D:/compCarsThesisData/data/misc/attributes.txt"
# car_type = "D:/compCarsThesisData/data/misc/car_type.mat"
# make_model_name = "D:/compCarsThesisData/data/misc/make_model_name.mat"
# carParts = "D:/compCarsThesisData/data/part/"

IMG_SIZE = 50
training_data = []
testing_data = []
MODEL_NAME = 'Classification'
LR = 1e-3




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def labelling(mylabel):
  if mylabel == '-1':
    return -1 #'uncertain'
  elif mylabel == '1':
    return 1 #'Front'
  elif mylabel == '2':
    return 2 #'Rear'
  elif mylabel == '3':
    return 3 #'Side'
  elif mylabel == '4':
    return 4 #'Front-Side'
  elif mylabel == '5':
    return 5 #'Read-Side'

for root, _, files in os.walk(image):
  cdp = os.path.abspath(root)

  for f in files:
    name,ext = os.path.splitext(f)
    if ext == ".jpg":
      cip = os.path.join(cdp, f)
      ci = mpg.imread(cip)
      ci = rgb2gray(ci)
      # images = cv2.resize(ci,(IMG_SIZE,IMG_SIZE))

      # images = rgb2gray(images)
      images = cv2.cv2.resize(ci,(IMG_SIZE,IMG_SIZE))
      images = np.array(images)
      # print(images.shape)
      #images = images / 255
      # print(images)
      # count = count + 1 
      label = cip.replace('image','label')
      label = label.replace('.jpg', '.txt')
   

      lines =[line.rstrip('\n') for line in open(label)]
      # print(lines[0])
      lines = np.array(lines)
      # print(lines)


      training_data.append((images,labelling(lines[0])))
      shuffle(training_data)
      np.save('training_data.npy',training_data)
      # print(training_data)


for img in tqdm(os.listdir(image)):
  path = os.path.join(image,img)
  #img_num = img.split('.')[0]
  # img_num = img_num.split('.')[0]
  testing_data.append((np.array(images),np.array(labelling(lines[0]))))
  shuffle(testing_data)
      # print(testing_data)

  train = training_data[:-5]
  # print(train)
  test = testing_data[-5:]
# print("train-Test",train,test)
  X_train = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        # print(X_train.shape)
  y_train = [i[1] for i in training_data]
        # print(y_train)
  X_test = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  print("XTEST",X_test)
  y_test = [i[1] for i in testing_data]
  print(y_test)

# print("XTRAIN---------------------------------------------------------------------",y_train)

tf.reset_default_graph()
convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')
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
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#({'input': X_test}, {'targets': y_test})
