import scipy.io as sio 
import numpy as np 
import resnet3 as res
import cv2
import aug

def getAverageVector(inp):
	a = inp/np.linalg.norm(inp,axis=1,keepdims=True)
	a = np.sum(a,axis=0)
	a = a/np.linalg.norm(a)
	return a

def processImg(img):
	img = cv2.resize(img,(122,144))
	M2 = np.float32([[1,0,11],[0,1,0]])
	img = cv2.warpAffine(img,M2,(144,144))
	img = img[8:136,8:136]
	return img

def getImglist():
	d = {}
	f = open('novel1.txt')
	for i in f:
		i = i.strip()
		lb = i.split('\\')[-2]
		if not lb in d:
			d[lb]=[]
		d[lb].append(i)
	return d

def main():
	print('Reading list...')
	d = getImglist()
	print('Dict size:',len(d))
	lbs = []
	dts = []
	cnt = 0
	for k in d:
		cnt+=1
		print(cnt)
		lst = d[k]
		lbs.append(k)
		imgs = []
		for i in lst:
			# img = processImg(cv2.imread(i,1))
			img = cv2.imread(i,1)
			imgs.append(img)
		#do augmentation
		imgs = aug.process(imgs)
		imgs = np.float32(imgs)
		print(imgs.shape)
		reall = []
		re = res.eval(imgs)
		# print(re.shape)
		print(re.shape)
		dts.append(getAverageVector(re))
	# dts = np.concatenate(dts,axis=0)
	dts = np.float32(dts)
	print(dts.shape)
	sio.savemat('novel1.mat',{'data':dts,'label':lbs})

def gettest():
	f = open('set1list.txt')
	lbs = []
	dts = []
	cnt = 0
	imgs = []
	for i in f:
		cnt+=1
		if cnt%10==0:
			print(cnt)
		i = i.strip()
		lb = i.split('\\')[-2]
		lbs.append(lb)
		img = processImg(cv2.imread(i,1))
		imgs.append(img)
		# img = cv2.imread(i,1)
		# img = aug.process([img])
		# img = np.float32(img)
		# print(img.shape)
		# re = res.eval(img)
		# dts.append(getAverageVector(re))
		# img = img.reshape([1,128,128,3])
	numb = len(imgs)//100
	for i in range(numb-1):
		print('Pro',i*100)
		re = res.eval(imgs[i*100:i*100+100])
		dts.append(re)
		print(re.shape)
	re = res.eval(imgs[(numb-1)*100:])
	print(re.shape)
	dts.append(re)
	dts = np.concatenate(dts,axis=0)
	# dts = np.float32(dts)
	print(dts.shape)
	# dts = np.float32(dts).reshape([-1,1024])
	sio.savemat('set1data_res50.mat',{'data':dts,'label':lbs})
	# res.closesession()

gettest()
# main()