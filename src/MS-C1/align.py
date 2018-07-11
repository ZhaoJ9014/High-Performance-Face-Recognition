import pandas as pd 
import numpy as np 
import cv2 
import os

def get4pts(pts):
	# pts = pts.split(' ')
	pts = [float(i) if float(i)>0 else 0 for i in pts]
	a = pts[0::2]
	b = pts[1::2]
	# print (a, b)
	# input()
	pts = np.float32(list(zip(a,b)))
	le = (pts[1]+pts[2])/2
	re = (pts[3]+pts[4])/2
	mse = (pts[5]+pts[6])/2
	nse = pts[0]
	return np.float32([le,re,nse,mse])

def get_010(img,lmk,img_name, save_path):
	M = np.float32([[1,0,2000],[0,1,2000]])
	rows,cols,_ = img.shape
	img = cv2.warpAffine(img,M,(cols+4000,rows+4000))
	lmk = lmk + np.float32([[2000,2000]])
	nse_c = lmk[2]
	leye_c = lmk[0]
	reye_c = lmk[1]
	eye_c = (lmk[0]+lmk[1])/2
	mse_c = lmk[3]
	flip = False
	# if nse_c[0]<mse_c[0]:
	# 	flip = True
	# else:
	# 	flip = False
	
	facey = eye_c[1] - (mse_c[1] - eye_c[1])*(49/45.)
	facey2 = mse_c[1] + (mse_c[1] - eye_c[1])*(49/45.)
	facex = (mse_c[0] + eye_c[0])/2 - (mse_c[1] - eye_c[1])*(61/45.)
	facex2 = (mse_c[0] + eye_c[0])/2 + (mse_c[1] - eye_c[1])*(61/45.)
	faceimg = img[int(facey):int(facey2),int(facex):int(facex2)]

	faceimg = cv2.resize(faceimg,(122,144))
	cv2.imshow('faceimg', faceimg)
	cv2.waitKey(0)
	cv2.imwrite(save_path+img_name,faceimg)

def rotate(img,lmk):
	leye_c = lmk[0]
	reye_c = lmk[1]
	eye_c = (leye_c + reye_c)/2
	mse_c = lmk[3]
	lmk2 = lmk-mse_c.reshape([-1,2])
	kaku = eye_c - mse_c
	kaku = np.arctan(-kaku[0]/kaku[1])/np.pi*180.0
	# print(kaku)
	rows,cols,_ = img.shape
	M = cv2.getRotationMatrix2D((int(mse_c[0]),int(mse_c[1])),kaku,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	kaku = kaku*np.pi / 180.0
	m2 = np.float32([[np.cos(kaku),-np.sin(kaku)],[np.sin(kaku),np.cos(kaku)]])
	lmk2 = lmk2.dot(m2)
	lmk2 += mse_c
	return dst,lmk2

def processFile(path,ptsstr):
	img = cv2.imread(path)
	dstpath = path.replace('IJB-A_images','parts')
	if not os.path.exists(os.path.dirname(dstpath)):
		os.makedirs(os.path.dirname(dstpath))
	pts = get4pts(ptsstr)
	img,pts = rotate(img,pts)
	get_010(img,pts,path)

# fout = open('avail_imgs.txt','w')
path = 'O:\\[FY2017]\\IJBA\\IJBA_Original_dataset\\IJB-A_images\\'
f_in = open('res.txt')
save_path = 'O:\\[FY2017]\\IJBA\\data4\\'

points = np
for line in f_in:
	line = line.strip()
	# print (line)

	temp = line.split(' ')
	img_name = temp[0]
	img_path = path + img_name
	img = cv2.imread(img_path)
	# print (img_path)

	try:
		pts = get4pts(temp[1:])
		img2,pts = rotate(img,pts)
		get_010(img2,pts,img_name, save_path)
	except:
		continue