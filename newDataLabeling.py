import os 
import numpy as np
import matplotlib.image as mpg
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
from sklearn import preprocessing

# path = "D:/compCarsThesisData/data/image/"
pathfor1000Samples = "D:/Morethan1000samples/data/image/"

# train = open("D:/compCarsThesisData/data/train_test_split/classification/Training_mostFrequentData.txt", "r")

# test = open("D:/compCarsThesisData/data/train_test_split/classification/Testing_mostFrequentData.txt","r")

# num_classes = 2004
training_data = []
training_labels = []
# testing_data = []
# testing_labels = []
count = 0
# for x in train:
  #  abspath = str(path+x).rstrip("\n\r")
   # print(abspath)
   # print(abspath.rsplit('/')[5])
for root, _, files in os.walk(pathfor1000Samples):
  cdp = os.path.abspath(root)

  for f in files:
    name, ext = os.path.splitext(f)

    if ext == ".jpg":
      cip = os.path.join(cdp,f)
      read_image = mpg.imread(cip)
      resize_image = cv2.cv2.resize(read_image,(224,224))
      read_image_float16 = resize_image.astype('float16')
      resize_image_preprocess = read_image_float16 / 255.0
      training_data.append(resize_image_preprocess) 
      count = count + 1
      print(count)
      import gc
      gc.collect()
   
      training_labels.append(int(cdp.split('\\')[4]))
      # print(training_labels)
      # exit()

print(len(training_labels))
# train.close()

  #  print(labels)
  #  print(training_labels)  


np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan1000samplesData/Training_Data_1000Samples',training_data)
np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan1000samplesData/Training_Labels_1000Samples',training_labels)
# print("Successfully Saved...! Training Data And Labels")

# del training_data, training_labels

# print("Building Testing Data Now ...")

# for y in test:

#   abspath_test = str(path + y).rstrip("\n\r")
#   print(abspath_test)
  
#   read_image_test = mpg.imread(abspath_test)
#   read_image_test_resized = cv2.cv2.resize(read_image_test,(224,224))
#   read_image_float16_test = read_image_test_resized.astype('float32')
#   read_image_test_resized_preprocess = read_image_float16_test / 255.0

#   testing_data.append(read_image_test_resized_preprocess)
#   testing_labels.append(y.rsplit('/')[0])
#   # print(test_labels)
#   # print(testing_labels)

# test.close()
# import gc
# gc.collect()

# np.save('D:/Inception_preprocessed_data_Labels_2004/Top5/TestingData_Top5',testing_data)
# np.save('D:/Inception_preprocessed_data_Labels_2004/Top5/TestingLabels_Top5',testing_labels)

# plt.imshow(testing_data[1900])
# plt.show()

# print(len(testing_labels))
# print("Successfully saved...! Testing Data and Labels")

# print("Aborting.... Enjoy Machine Learning You A**HOLE")
# del testing_data, testing_labels