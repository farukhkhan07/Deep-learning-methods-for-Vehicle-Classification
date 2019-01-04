from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import scipy.io as cio
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpg
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2

a = cio.loadmat("D:/compCarsThesisData/data/misc/make_model_name.mat")

images = "D:\compCarsThesisData\data\image"
# images = "D:/practiceData/data/image/"
IMG_SIZE = 64
MODEL_NAME = 'Classification'
LR = 1e-3

b = a['make_names']
# c = b.reshape(163,)
d = []
count = 0
for i in range(b.size):
  d.append(b[i][0][0])
  #print(d)

labels_dic = {v: k for v, k in enumerate(d)}
print(labels_dic)
# indices = np.arange(163)
# depth = 163

# y = tf.one_hot(indices,depth)
# # print(y)

# sess = tf.Session()

# result = sess.run(y)
# print(result.shape)
# labels = []

# labels.append((result,labels_dic))
# print(labels)
image1 = []
label1 = []
# pre = 1
# current = 0
for root, _, files in os.walk(images):
  cdp = os.path.abspath(root)

  for f in files:
    
    name,ext = os.path.splitext(f)
    boolean = True
     
    if ext == ".jpg":
      cip = os.path.join(cdp,f)      
      ci = mpg.imread(cip)
      image = cv2.cv2.resize(ci,(IMG_SIZE,IMG_SIZE))
      np.array(image1.append(image))
      # For ALL DATA
      # arr = np.zeros(163)
      arr = np.zeros(163)
      print(cip.split('\\')[4])
      arr[int(cip.split('\\')[4])-1] = 1

      # exit()
      label1.append(arr)
      count = count+1
      print(count)

print("done :)")

# np.save('compCarsData',image1)
# np.save('labelData',label1)
# image2 = np.load('compCarsData.npy')
# label2 = np.load('labelData.npy')
training_data = []

training_data.append([np.array(image1),np.array(label1)])
#print("TrainingData",training_data)
shuffle(training_data)
# np.save('training_data_with_One_Hot', training_data)

testing_data = []
testing_data.append([np.array(image1),np.array(label1)])
#print("TestingDATA",testing_data)
# np.save('testing_data_with_One_Hot',testing_data)
shuffle(testing_data)

# training_data = np.load('training_data_with_One_Hot.npy')
# testing_data = np.load('testing_data_with_One_Hot.npy')

# print("Training Data Shape",training_data)
print(testing_data[0][1].shape)
train = training_data[-2000:]
test = testing_data[-2000:]

print("TRAINS",train[0][0].shape)
print("TEST SHAPE",test[0][0][:1].shape)
X_train = train[0][0] #np.array(train[0]).reshape(None, IMG_SIZE, IMG_SIZE, 3)

print("XTRAINING DATA",X_train.shape)
y_train = train[0][1]#np.array([i[0][1] for i in train])
print("Ytrain",y_train.shape)
X_test = test[0][0]
print("XTEST", X_test.shape) #np.array([i[0] for i in test]).reshape(None, IMG_SIZE, IMG_SIZE, 3)
y_test = test[0][1] #np.array([i[1] for i in test])
print("YTEST",y_test.shape)

# sess = tf.Session()cls


tf.reset_default_graph()
convnet = input_data(shape=(None,IMG_SIZE,IMG_SIZE,3),name='input')
#shape=[None, IMG_SIZE, IMG_SIZE, 1],
convnet = conv_3d(convnet, 32, 5, activation='relu')
convnet = max_pool_3d(convnet, 5)
convnet = conv_3d(convnet, 64, 5, activation='relu')
convnet = max_pool_3d(convnet, 5)
convnet = conv_3d(convnet, 128, 5, activation='relu')
convnet = max_pool_3d(convnet, 5)
convnet = conv_3d(convnet, 64, 5, activation='relu')
convnet = max_pool_3d(convnet, 5)
convnet = conv_3d(convnet, 32, 5, activation='relu')
convnet = max_pool_3d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 163, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


#({'input': X_test}, {'targets': y_test})
# sess.run()cls


# b = a['make_names']
# b = np.asarray(b)

# print(b)

# onehotencoder = OneHotEncoder(categorical_features= [0])

# b = onehotencoder.fit_transform(b).toarray()
