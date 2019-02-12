import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpg
import numpy as np
import os

N = 0
training_labels = []
# bbox = "/run/media/fdai5182/LAMA MADAN/Morethan4000samples/data/labels"
# imagepath = "/run/media/fdai5182/LAMA MADAN/Morethan4000samples/data/image"

bbox = "D:/Morethan4000samples/data/labels"
imagepath = "D:/Morethan4000samples/data/image"

for root, _, files in os.walk(imagepath):
        cdp = os.path.abspath(root)
        for f in files:
                name, ext = os.path.splitext(f)
                if ext == ".jpg":
                        cip = os.path.join(cdp,f)
                        N += 1	
            
print(N) 

imageX = np.zeros((N,300,300,3), dtype='float32')

i = 0

for root, _ , files in os.walk(imagepath):
        cdp = os.path.abspath(root)
        print(cdp)
        for f in files:
                ct = 0
                name, ext = os.path.splitext(f)
                if ext == ".jpg":
                        cip = os.path.join(cdp,f)
                        read = mpg.imread(cip)
                        cipLabel = cip.replace('image','labels')
                        cipLabel = cipLabel.replace('.jpg','.txt')
                        nameL , extL = os.path.splitext(cipLabel)
                        if extL == '.txt':
                                boxes = open(cipLabel, 'r')
                                for q in boxes:
                                        ct = ct + 1 
                                        if ct == 3:
                                                x1 = int(q.rsplit(' ')[0])
                                                y1 = int(q.rsplit(' ')[1])
                                                x2 = int(q.rsplit(' ')[2])
                                                y2 = int(q.rsplit(' ')[3])
                                                readimage = read[y1:y2, x1:x2]
                                                resize = cv2.cv2.resize(readimage,(300,300))
                                                imageX[i] = resize
                        training_labels.append(int(cip.split('\\')[4]))
                        i += 1	
                        print(i)


imageX /= 255.0
print(imageX.shape)
plt.imshow(imageX[18000])
plt.show()


np.save("D:/Inception_preprocessed_data_Labels_2004/Morethan4000samplesData/training_images", imageX)
np.save("D:/Inception_preprocessed_data_Labels_2004/Morethan4000samplesData/trainin_labels",training_labels)
    
	   	
