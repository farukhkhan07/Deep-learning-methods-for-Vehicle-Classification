import os
import matplotlib.pyplot as plt
import matplotlib.image as mpg
import cv2
import gc
import numpy as np
# bbox = "D:/Morethan1000samples/data/labels/"
# imagepath = "D:/Morethan1000samples/data/image/"

# bbox = "D:/Morethan1000samples/data/labels/"
# imagepath = "D:/Morethan1000samples/data/image/"

bbox = "D:/Morethan2000samples/data/labels/"
imagepath = "D:/Morethan2000samples/data/image/"

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
								# print(read_img.shape)
								read_img_bbox = read_img[y1:y2, x1:x2,:]
								# print(read_img_bbox.shape)
								resize_img = cv2.cv2.resize(read_img_bbox,(300,300))
								resize_img_pre = resize_img / 255.0
								# plt.imshow(resize_img)
								# plt.show()
								# print(resize_img.shape)
								# exit()
								# training_data.append(resize_img)
								# training_labels.append(int(cipimg.split('\\')[4]))
								training_data.append(resize_img_pre)
								training_labels.append(int(cipimg.split('\\')[4]))
								# for i in range(37529):
								i = 0
								if len(training_data) >= 10000 and len(training_labels) >= 10000:
									np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_Data_2000Samples_chunk'+ str(i),training_data)
									np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan2000samplesData/Training_labels_2000Samples_chunk' + str(i),training_labels)
									training_data = []
									training_labels = []
									i = i + 1	
							except Exception as e:
								print(str(e), cip)	
							count = count + 1
							print(count)	




# np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan1000samplesData/Training_Data_1000Samples',training_data)
# np.save('D:/Inception_preprocessed_data_Labels_2004/Morethan1000samplesData/Training_Labels_1000Samples',training_labels)

