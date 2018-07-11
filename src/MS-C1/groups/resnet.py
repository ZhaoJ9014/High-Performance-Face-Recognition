import tensorflow as tf 
import numpy as np 
import model as M 
import cv2

CLASS = 20000
# CLASS = 24979
# CLASS = 15572
BSIZE = 128

def block(mod,output,stride):
	inp = mod.get_current()[1][-1]
	aa = mod.get_current()
	if inp==output:
		if stride==1:
			l0 = mod.get_current()
		else:
			l0 = mod.maxpoolLayer(stride)
	else:
		l0 = mod.convLayer(1,output,activation=M.PARAM_RELU,batch_norm=True,stride=stride)
	mod.convLayer(1,output//4,activation=M.PARAM_RELU,batch_norm=True,layerin=aa,stride=stride)
	mod.convLayer(3,output//4,activation=M.PARAM_RELU,batch_norm=True)
	mod.convLayer(1,output,batch_norm=True)
	mod.sum(l0)
	mod.activate(M.PARAM_RELU)
	return mod

def res_18():
	with tf.name_scope('imginput'):
		imgholder = tf.placeholder(tf.float32,[None,224,224,3])
		# img2 = tf.image.resize_images(imgholder,(224,224))
		img2 = imgholder
	with tf.name_scope('labholder'):
		labholder = tf.placeholder(tf.int64,[None,CLASS])
		print(labholder)
	mod = M.Model(img2,[None,224,224,3])
	mod.set_bn_training(False)
	mod.convLayer(7,64,activation=M.PARAM_RELU,stride=2,batch_norm=True)
	mod.maxpoolLayer(3,stride=2)
	# 3x256
	block(mod,256,1)
	block(mod,256,1)
	block(mod,256,1)
	# 4x512
	block(mod,512,2)
	block(mod,512,1)
	block(mod,512,1)
	block(mod,512,1)
	# 6x1024
	block(mod,1024,2)
	block(mod,1024,1)
	block(mod,1024,1)
	block(mod,1024,1)
	block(mod,1024,1)
	block(mod,1024,1)
	# 3x2048
	block(mod,2048,2)
	block(mod,2048,1)
	block(mod,2048,1)
	mod.avgpoolLayer(7)
	mod.flatten()
	featurelayer = mod.get_current_layer()
	with tf.variable_scope('enforced_layer'):
		classlayer,evallayer = M.enforcedClassfier(featurelayer,2048,labholder,BSIZE,CLASS,dropout=1,enforced=False)
	# mod.fcLayer(20000)
	# classlayer = mod.get_current_layer()
	# evallayer = classlayer
	return classlayer,imgholder,labholder,featurelayer,evallayer

with tf.variable_scope('MainModel'):
	classlayer,imgholder,labholder,featurelayer,evallayer = res_18()

modelpath = './part6/'
modelname = 'Epoc15Iter1579.ckpt'

def evaluate():
	import cv2
	# listf = open('set1list.txt')
	listf = open('1k5_set2.txt')
	imglist = []
	for i in listf:
		imglist.append(i.replace('\n',''))
	with tf.Session() as sess:
		M.loadSess(modelpath,sess=sess,modpath=modelpath+modelname)
		res = []
		imgs = []
		print('reading images...')
		for pic in imglist:
			img = cv2.imread(pic,1)
			img = cv2.resize(img,(122,144))
			M2 = np.float32([[1,0,11],[0,1,0]])
			img = cv2.warpAffine(img,M2,(144,144))
			img = cv2.flip(img,1)
			img = np.float32(img)[8:136,8:136]
			img = cv2.resize(img,(224,224))
			imgs.append(img)
		splits = len(imgs)//100
		print('Splits in total...')
		for i in range(splits-1):
			cl = sess.run(tf.nn.softmax(evallayer),feed_dict={imgholder:imgs[i*100:i*100+100]})
			print(cl.shape)
			res.append(cl)
		cl = sess.run(tf.nn.softmax(evallayer),feed_dict={imgholder:imgs[(splits-1)*100:]})
		print(cl.shape)
		res.append(cl)
		res = np.concatenate(res,axis=0)
		print(res.shape)
		import scipy.io as sio 
		sio.savemat('enf56_1.mat',{'data':res})
		fout = 'testresult.txt'
		fout = open(fout,'w')
		for i in res:
			fout.write(str(i)+'\n')
		fout.close()
evaluate()