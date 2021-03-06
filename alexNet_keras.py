import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils
from random import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpg
import cv2
np.random.seed(1000)

# (2) Get Data
# import tflearn.datasets.oxflower17 as oxflower17
# x, y = oxflower17.load_data(one_hot=True)

X_train = np.load('D:/Inception_preprocessed_data_Labels_2004/Morethan4000samplesData/250250/training_images.npy')#('D:/ThesisWork/S_224_Training_data.npy')#training_images
X_test = np.load('D:/Inception_preprocessed_data_Labels_2004/Morethan4000samplesData/250250/trainin_labels.npy')#('D:/ThesisWork/S_224_Training_labels.npy')#training_labels
print(X_test)
# exit()
# plt.imshow(X_train[1500])
# plt.show()
# X_train = X_train/255.0
le = preprocessing.LabelEncoder()
le.fit(X_test)
transform_trainLabels = le.transform(X_test)

train_labels_hotEncode = np_utils.to_categorical(transform_trainLabels,len(set(transform_trainLabels)))
print(X_train.shape)
print(train_labels_hotEncode.shape)
shuffle(X_train)
shuffle(train_labels_hotEncode)
# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(250,250,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(250*250*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()
# ndm = keras.optimizers.nadam(lr = 0.0001, epsilon=1e-08)
# keras.optimizers.nadam()
# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# (5) Train
model.fit(X_train, train_labels_hotEncode, batch_size=32, epochs=50, verbose=1,validation_split=0.3, shuffle=True)# shuffle=True

predict = mpg.imread("D:/Morethan4000samples/testdata/39/1367/2007/1e39f2a877b507.jpg")
predict_resize = cv2.cv2.resize(predict,(250,250))
mod = model.predict(predict)
print(mod.shape)
print(mod)

mod = np.argmax(mod,axis=1)
train_labels_hotEncode = np.argmax(train_labels_hotEncode,axis=1)
cm = confusion_matrix(train_labels_hotEncode, mod)

print(cm)
plt.imshow(cm)
plt.show()
