import cv2
import numpy as np 

PIX = 144

def getShiftMat(shiftNumX,shiftNumY):
	return np.float32([[1,0,shiftNumX],[0,1,shiftNumY]])

def getScaleAdjstMat(scale):
	shiftNum = PIX*(1-scale)/2
	return getShiftMat(shiftNum,shiftNum)

def getBluredImg(img,blurNum):
	img2 = cv2.blur(img,(blurNum,blurNum))
	return img2

def getScaledImg(img,scale):
	img2 = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
	img2 = cv2.warpAffine(img2,getScaleAdjstMat(scale),(PIX,PIX))
	return img2

def getRotateImg(img,degree):
	M = cv2.getRotationMatrix2D((PIX/2,PIX/2),degree,1)
	img2 = cv2.warpAffine(img,M,(PIX,PIX))
	return img2

def getFlipImg(img):
	img2 = cv2.flip(img,1)
	return img2

def contrast(img,F):
	img = np.float32(img)
	# F = 259*(ind+255)/(255*(259-ind))
	img = F*(img-128)+128
	img[img>254]=254.0
	img[img<0]=0.0
	img = img.astype(np.uint8)
	return img

def bright(img,ind):
	img = np.float32(img)
	img += ind
	img[img>254]=254.0
	img[img<0]=0.0
	img = img.astype(np.uint8)
	return img

#rotate 10 degree
#flip
#contra 1.1 0.9

def goContra(lst):
	lst2 = []
	for i in lst:
		lst2.append(i)
		lst2.append(contrast(i,0.7))
		lst2.append(bright(i,80))
	return lst2

def goFlip(lst):
	lst2 = []
	for i in lst:
		lst2.append(i)
		lst2.append(getFlipImg(i))
	return lst2

def goRorate(lst):
	lst2 = []
	for i in lst:
		lst2.append(i)
		lst2.append(getRotateImg(i,10))
		lst2.append(getRotateImg(i,-10))
	return lst2

def randomcrop(lst,crop):
	import random
	lst2 = []
	for img in lst:
		for i in range(crop):
			w = random.randint(0,16)
			h = random.randint(0,16)
			lst2.append(img[w:w+128,h:h+128])
	return lst2

def processImg(img):
	img = cv2.resize(img,(122,144))
	M2 = np.float32([[1,0,11],[0,1,0]])
	img = cv2.warpAffine(img,M2,(144,144))
	# img = img[8:136,8:136]
	return img

def postprocess(img):
	img = img[8:136,8:136]
	return img

def process(img):
	# print(img[0].shape)
	a = [processImg(i) for i in img]
	# a = goRorate(a)
	a = goFlip(a)
	# a = goContra(a)
	# a = [postprocess(i) for i in a]
	a = randomcrop(a,20)
	a = np.float32(a)
	return a