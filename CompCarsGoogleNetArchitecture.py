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

# path_to_images = "D:/compCarsThesisData/data/images/"
# train_path = "D:/compCarsThesisData/data/train_test_split/classification/train.text"
# test_path = "D:/compCarsThesisData/data/train_test_split/classification/test.txt"

# train_split = open(train_path,"r")
# test_split = open(test_path,"r")
# img_rows = 224
# img_cols = 224

# for x in train_split:
#   abspath_train = str(path_to_images+train_split).rstrip("\n\r")

#   train_image_read = mpg.imread(abspath_train)

#   train_image = cv2.cv2.resize(train_image_read,(img_rows,img_cols))

#   X_train_images = np.array([train_image for train_image in X_train_images[:,:,:,:]])



# for y in train_split: 
#   abspath_test = str(path_to_images+test_split).rstrip("\n\r")

#   test_image_read = mpg.imread(abspath_test)

#   test_image = cv2.cv2.resize(test_image_read,(img_rows,img_cols))

#   Y_test_images = np.array([test_image for test_image in Y_test_images[:,:,:,:]])

# X_train = np.load('D:/ThesisWork/TrainingImages_googleNet.npy')
# X_test = np.load('D:/ThesisWork/TrainingImagesLabels_googleNet.npy')
# y_train = np.load('D:/ThesisWork/TestingImages_googleNet.npy')
# y_test = np.load('D:/ThesisWork/TestingImagesLabels_googleNet.npy')

X_train = np.load('D:/ThesisWork/224_Training_data.npy')#training_images
X_test = np.load('D:/ThesisWork/224_Training_labels.npy')#training_labels
y_train = np.load('D:/ThesisWork/224_Testing_data.npy')#testing_images 
y_test = np.load('D:/ThesisWork/224_Testing_labels.npy')#testing_labels

# X_train = X_train / 255.0
# y_train = y_train / 255.0

print("X_Training" ,X_train.shape)
print("X_TEST", X_test.shape)
print("Y_train", y_train.shape)
print("y_test", y_test.shape)



def inception_module(image, 
            filters_1x1, 
            filters_3x3_reduce, 
            filter_3x3,
            filters_5x5_reduce,
            filters_5x5, 
            filters_pool_proj, 
            name=None):
  conv_1x1 = Conv2D(filters_1x1, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer= bias_init)(image)
  conv_3x3 = Conv2D(filters_3x3_reduce, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer= bias_init)(image)
  conv_3x3 = Conv2D(filter_3x3,(3,3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

  conv_5x5 = Conv2D(filters_5x5_reduce,(1,1), padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer= bias_init)(image)
  conv_5x5 = Conv2D(filters_5x5, (3,3), padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

  pool_proj = MaxPool2D((3,3), strides=(1,1), padding='same')(image)
  pool_proj = Conv2D(filters_pool_proj, (1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer= bias_init)(pool_proj)

  output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

  return output
  
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(200,200,3))

image = Conv2D(64,(7,7),padding='same', strides=(2,2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)

image = MaxPool2D((3,3), padding='same', strides=(2,2), name='max_pool_1_3x3/2')(image)
image = Conv2D(64, (1,1), padding='same', strides=(1,1), activation='relu', name='conv_2a_3x3/1' )(image)
image = Conv2D(192, (3,3), padding='same', strides=(1,1), activation='relu', name='conv_2b_3x3/1')(image)
image = MaxPool2D((3,3), padding='same', strides=(2,2), name='max_pool_2_3x3/2')(image)

image = inception_module(image,
                    filters_1x1= 64,
                    filters_3x3_reduce= 96,
                    filter_3x3 = 128,
                    filters_5x5_reduce=16,
                    filters_5x5= 32,
                    filters_pool_proj=32,
                    name='inception_3a')

image = inception_module(image,
                            filters_1x1=128,
                            filters_3x3_reduce=128,
                            filter_3x3=192,
                            filters_5x5_reduce=32,
                            filters_5x5=96,
                            filters_pool_proj=64,
                            name='inception_3b')

image = MaxPool2D((3,3), padding='same', strides=(2,2), name='max_pool_3_3x3/2')(image)

image = inception_module(image, 
                            filters_1x1=192,
                            filters_3x3_reduce=96,
                            filter_3x3=208,
                            filters_5x5_reduce=16,
                            filters_5x5=48,
                            filters_pool_proj=64,
                            name='inception_4a')

image1 = AveragePooling2D((5,5), strides=3)(image)
image1 = Conv2D(128, (1,1), padding='same', activation='relu')(image1)
image1 = Flatten()(image1)
image1 = Dense(1024, activation='relu')(image1)
image1 = Dropout(0.7)(image1)
image1 = Dense(163, activation='softmax', name='auxilliary_output_1')(image1)

image = inception_module(image,
                            filters_1x1 = 160,
                            filters_3x3_reduce= 112,
                            filter_3x3= 224,
                            filters_5x5_reduce= 24,
                            filters_5x5= 64,
                            filters_pool_proj=64,
                            name='inception_4b')

image = inception_module(image,
                           filters_1x1= 128,
                           filters_3x3_reduce = 128,
                           filter_3x3= 256,
                           filters_5x5_reduce= 24,
                           filters_5x5=64,
                           filters_pool_proj=64,
                           name='inception_4c')

image = inception_module(image,
                           filters_1x1=112,
                           filters_3x3_reduce=144,
                           filter_3x3= 288,
                           filters_5x5_reduce= 32,
                           filters_5x5=64,
                           filters_pool_proj=64,
                           name='inception_4d')

image2 = AveragePooling2D((5,5), strides=3)(image)
image2 = Conv2D(128, (1,1), padding='same', activation='relu')(image2)
image2 = Flatten()(image2)
image2 = Dense(1024, activation='relu')(image2)
image2 = Dropout(0.7)(image2)
image2 = Dense(163, activation='softmax', name='auxilliary_output_2')(image2)
  
image = inception_module(image,
                            filters_1x1=256,
                            filters_3x3_reduce=160,
                            filter_3x3=320,
                            filters_5x5_reduce=32,
                            filters_5x5=128,
                            filters_pool_proj=128,
                            name= 'inception_4e')

image = MaxPool2D((3,3), padding='same', strides=(2,2), name='max_pool_4_3x3/2')(image)

image = inception_module(image,
                           filters_1x1=256,
                           filters_3x3_reduce=160,
                           filter_3x3= 320,
                           filters_5x5_reduce=32,
                           filters_5x5= 128,
                           filters_pool_proj=128,
                           name='inception_5a')

image = inception_module(image, 
                           filters_1x1=384,
                           filters_3x3_reduce=192,
                           filter_3x3=384,
                           filters_5x5_reduce=48,
                           filters_5x5=128,
                           filters_pool_proj=128,
                           name='inception_5b')

image = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(image)

image = Dropout(0.4)(image)
image = Dense(163, activation='softmax', name='output')(image)

model = Model(input_layer, [image,image1,image2], name='inception_v1')

model.summary()

epochs = 250
initial_lrate = 0.01

def decay(epoch, steps=100):
  initial_lrate = 0.01
  drop = 0.96
  epochs_drop = 8
  lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))#
  return lrate

# sgd = SGD(lr = initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay)

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'],loss_weights=[1,0.3,0.3], optimizer='sgd', metrics=['accuracy'])
# 
history = model.fit(X_train, [X_test,X_test,X_test], validation_data=(y_train,[y_test,y_test,y_test]), epochs=epochs, batch_size= 32, callbacks=[lr_sc])

import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['output_acc'])
plt.plot(history.history['val_output_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc = 'upper left')
plt.show()


predictingimage = "D:/compCarsThesisData/data/image/108/1531/2012/74bf97739e9260.jpg" #67/1698/2010/6805eb92ac6c70.jpg"
predictImageRead = mpg.imread(predictingimage)

resizingImage = cv2.cv2.resize(predictImageRead,(200,200))
reshapedFinalImage = np.expand_dims(resizingImage, axis=0)
npimage = np.asarray(reshapedFinalImage)
m = model.predict(npimage)
print(m)
n = np.argmax(m,axis=-1)
plt.imshow(n)
plt.show()