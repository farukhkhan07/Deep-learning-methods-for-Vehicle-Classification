import os
import matplotlib.pyplot as plt
import matplotlib.image as mpg
import cv2
import gc
import numpy as np
from sklearn.preprocessing import normalize
import gc
# bbox = "D:/Morethan1000samples/data/labels/"
# imagepath = "D:/Morethan1000samples/data/image/"

# bbox = "D:/Morethan1000samples/data/labels/"
# imagepath = "D:/Morethan1000samples/data/image/"

bbox = "D:/Morethan4000samples/data/labels/"
imagepath = "D:/Morethan4000samples/data/image/"

training_data = []
training_labels = []
count = 0


for root, _, files in os.walk(bbox):
	cdp = os.path.abspath(root)
	# print("Current Directory Label",cdp)
	for r , _ , t in os.walk(imagepath):
		cdpimg = os.path.abspath(r)	
		# print("Current Directory Image",cdpimg)
		for f in files:
			ct = 0
			name,ext = os.path.splitext(f)
			for s in t:
				n , e = os.path.splitext(s)
				if name == n and ext == ".txt" and e == ".jpg":
					cip = os.path.join(cdp,f)
					cipimg = os.path.join(cdpimg,s)
					txt = open(cip,"r")
					for q in txt:
						ct = ct + 1
						if ct == 3:
							x1 = int(q.rsplit(' ')[0])
							y1 = int(q.rsplit(' ')[1])
							x2 = int(q.rsplit(' ')[2])
							y2 = int(q.rsplit(' ')[3])
							# print(x1,y1,x2,y2)	
							try:
								read_img = mpg.imread(cipimg)
								read_img = read_img.astype('float32')
								# print(read_img.shape)
								read_img_bbox = read_img[y1:y2, x1:x2,:]
								# print(read_img_bbox.shape)
								resize_img = cv2.cv2.resize(read_img_bbox,(300,300))
								resize_img_pre = cv2.cv2.normalize(resize_img, None, dtype=cv2.CV_32F)
								# print(int(cipimg.split('\\')[4]))
								# exit()
								# resize_img_pre *= 255.0 / resize_img.max()
								# plt.imshow(resize_img_pre)
								# plt.show()
								# exit()
								# print(resize_img.shape)
								# exit()
								# training_data.append(resize_img)
								# training_labels.append(int(cipimg.split('\\')[4]))

								training_data.append(resize_img_pre)

								training_labels.append(int(cipimg.split('\\')[4]))
								gc.collect()
								# for i in range(37529):
								# 	if len(training_data) == 10000 and len(training_labels) == 10000:
								# 		np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_Data_2000Samples_chunk'+ str(i),training_data)
								# 		np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_labels_2000Samples_chunk' + str(i),training_labels)
								# 		print(len(training_data))
								# 		print(len(training_labels))
								# 		training_data = []
								# 		training_labels = []	
								# 		exit()
								
							except Exception as e:
								print("Error",str(e), cip)
							count = count + 1
							print(count)	
					txt.flush()
					txt.close()	



try:
	np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_Data_2000Samples',training_data)
	np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_Labels_2000Samples',training_labels)
except MemoryError as m:
	print("Cannot")

print("DONE")
