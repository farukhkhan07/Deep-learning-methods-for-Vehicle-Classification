import os 
import numpy as np
import matplotlib.image as mpg
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
from sklearn import preprocessing

path = "D:/compCarsThesisData/data/image/"

train = open("D:/compCarsThesisData/data/train_test_split/classification/trainsome.txt", "r")

test = open("D:/compCarsThesisData/data/train_test_split/classification/testsome.txt","r")

num_classes = 2004
training_data = []
training_labels = []
testing_data = []
testing_labels = []
count = 0
for x in train:
   abspath = str(path+x).rstrip("\n\r")
   # print(abspath)
   # print(abspath.rsplit('/')[5])

   read_image = mpg.imread(abspath)
   resize_image = cv2.cv2.resize(read_image,(224,224))
   resize_image_preprocess = resize_image / 255.0

   training_data.append(resize_image_preprocess) 
   import gc
   gc.collect()
   
   training_labels.append(x.rsplit('/')[1])
  #  print(training_labels)


train.close()

  #  print(labels)
  #  print(training_labels)  

np.save('D:/Inception_preprocessed_data_Labels_2004/Training_data_preprocessed_200x200',training_data)
np.save('D:/Inception_preprocessed_data_Labels_2004/Training_labels_KerasCompatibleFormat',training_labels)
print("Successfully Saved...! Training Data And Labels")

del training_data, training_labels

print("Building Testing Data Now ...")

for y in test:

  abspath_test = str(path + y).rstrip("\n\r")
  print(abspath_test)
  
  read_image_test = mpg.imread(abspath_test)
  read_image_test_resized = cv2.cv2.resize(read_image_test,(224,224))
  read_image_test_resized_preprocess = read_image_test_resized / 255.0

  testing_data.append(read_image_test_resized_preprocess)
  testing_labels.append(y.rsplit('/')[1])
  # print(test_labels)
  # print(testing_labels)

test.close()
import gc
gc.collect()

np.save('D:/Inception_preprocessed_data_Labels_2004/Testing_data_preprocessed_200x200',testing_data)
np.save('D:/Inception_preprocessed_data_Labels_2004/Testing_labels_preprocess_200x200',testing_labels)

plt.imshow(testing_data[1900])
plt.show()

print("Successfully saved...! Testing Data and Labels")

print("Aborting.... Enjoy Machine Learning You A**HOLE")
del testing_data, testing_labels