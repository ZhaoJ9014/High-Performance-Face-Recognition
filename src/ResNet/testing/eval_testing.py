import resnet2 as net
import numpy as np 
import cv2
import scipy.io as sio 
import os
from os import listdir
import random

def Average(inp):
	a = inp/np.linalg.norm(inp, axis=1, keepdims=True)
	a = np.sum(a, axis=0)
	a = a/np.linalg.norm(a)
	return a

path = r'O:\[FY2017]\MS-Challenges\code\evaluation_all\models\ensemble_google_submit3\test_data\5/'
path = path.replace('\\', '/')
print (path)

NUM = 0
imglist = path + 'dev5.txt'
with open(imglist) as f:
    NUM = sum(1 for _ in f)
f.close()

f = open(imglist, 'r')
res = []
imgs = []
labs = []
count = 0
n = 0
print('reading images...')
ff = []
for pic in f:
	n += 1
	pic = pic.strip()
	lab = pic.split('\\')[-1].replace('.jpg', '')
	# print (lab)
	img_path2 = pic
	try:
		img = cv2.imread(img_path2,1)
	except:
		count += 1
		ff.append(pic)
		continue
	img = cv2.resize(img,(122,144))
	M2 = np.float32([[1,0,11],[0,1,0]])
	img = cv2.warpAffine(img,M2,(144,144))
	
	imgs = []
	for i in range(20):
		w = random.randint(0, 16)
		h = random.randint(0, 16)
		img2 = img[w:w+128, h:h+128] 
		img2 = np.float32(img2)
		imgs.append(img2)

	for i in range(5):
		w = random.randint(0, 16)
		h = random.randint(0, 16)
		img2 = img[w:w+128, h:h+128] 
		img2 = cv2.flip(img2, 1)
		img2 = np.float32(img2)
		imgs.append(img2)

	feas = net.eval(imgs)
	feas_avg = Average(feas)
	feas_avg = np.array(feas_avg)
	labs.append(lab)
	res.append(feas_avg)
	if n % 10 == 0:
		print (str(n) + '/' + str(NUM) + '\t' + str(count))

save_path = path + 'dev5_Res.mat'
sio.savemat(save_path,{'data':res, 'label':labs})

f.close()