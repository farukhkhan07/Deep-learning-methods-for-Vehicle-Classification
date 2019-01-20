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
# AUTHOR: FARRUKH AHMED KHAN
# EMAIL : farukhcs15@gmail.com


IMG_SIZE = 224
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
  print(train_count, arr.shape, arr[77], abspath)
  # exit()

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
np.save('S_224_Training_data',training_images)
np.save('S_224_Training_labels', training_labels)
np.save('S_224_Testing_data', testing_images)
np.save('S_224_Testing_labels',testing_labels)

# =========================
# Data already Saved, Load it
# =========================
# X_train = np.load('D:/ThesisWork/Training_data.npy')#training_images
# y_train = np.load('D:/ThesisWork/Training_labels.npy')#training_labels
# X_test = np.load('D:/ThesisWork/Testing_data.npy')#testing_images
# y_test = np.load('D:/ThesisWork/Testing_labels.npy')#testing_labels
# print(X_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(y_train.shape)


exit()
with tf.device('/gpu:0'):
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
  convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy', name='targets')
  model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
  history = model.fit({'input': X_train}, {'targets': y_train}, n_epoch=20,
            validation_set=({'input': X_test}, {'targets': y_test}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())

# =========================
# For Saving The Model
# =========================
# model.save('my_trained_model.tflearn')
# np.save('training_finalized_data.npy', model)


predictingimage = "D:/compCarsThesisData/data/image/108/1531/2012/74bf97739e9260.jpg" #67/1698/2010/6805eb92ac6c70.jpg"
predictImageRead = mpg.imread(predictingimage)
resizingImage = cv2.cv2.resize(predictImageRead,(IMG_SIZE,IMG_SIZE))
reshapedFinalImage = np.expand_dims(resizingImage, axis=0)
# imagetoarray = np.array(resizingImage)
# reshapedFinalImage = imagetoarray.reshape(1,IMG_SIZE,IMG_SIZE,3)

# =========================
# For Prediction
# =========================
model_out = model.predict_label(reshapedFinalImage)
model_out1  = model.predict(reshapedFinalImage)
model_out2 = model_out1.argmax(axis=-1)
print(model_out.shape)
print(model_out)
print(model_out1)
# model_out_reshaped = model_out1.reshape(IMG_SIZE,IMG_SIZE,3)
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc = 'upper left')
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, model_out)
plt.title('Predicted confusion matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.imshow(cm, interpolation='nearest')
plt.show()


