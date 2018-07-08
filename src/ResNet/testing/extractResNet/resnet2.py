import tensorflow as tf 
import numpy as np 
import model as M 

CLASS = 66935
BSIZE = 128
blknum = 0

def block(mod,output,stride):
	global blknum
	with tf.variable_scope('block'+str(blknum)):
		inp = mod.get_current()[1][-1]
		aa = mod.get_current()
		if inp==output:
			if stride==1:
				l0 = mod.get_current()
			else:
				l0 = mod.maxpoolLayer(stride)
		else:
			l0 = mod.convLayer(1,output,activation=M.PARAM_RELU,stride=stride)
		mod.set_current_layer(aa)
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		mod.convLayer(1,output//4,activation=M.PARAM_RELU,batch_norm=True,stride=stride)
		mod.convLayer(3,output//4,activation=M.PARAM_RELU,batch_norm=True)
		mod.convLayer(1,output)
		mod.sum(l0)
		blknum+=1
	return mod

def res_18():
	with tf.name_scope('imginput'):
		imgholder = tf.placeholder(tf.float32,[None,128,128,3])
		img2 = tf.image.resize_images(imgholder,(224,224))
		# img2 = imgholder
	with tf.name_scope('labholder'):
		labholder = tf.placeholder(tf.int64,[None,CLASS])
		print(labholder)
	mod = M.Model(img2,[None,224,224,3])
	mod.set_bn_training(False)
	mod.convLayer(7,64,stride=2)
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
	print('check')
	mod.batch_norm()
	mod.dwconvLayer(7,1,pad='VALID')
	print(mod.get_current_layer())
	mod.flatten()
	featurelayer = mod.get_current_layer()
	# with tf.variable_scope('enforced_layer'):
	# 	classlayer,evallayer = M.enforcedClassfier2(featurelayer,2048,labholder,BSIZE,CLASS,dropout=1,enforced=True)
	return imgholder,featurelayer

with tf.variable_scope('MainModel'):
	imgholder,featurelayer = res_18()

sess = tf.Session()
# M.loadSess('./model2/',sess=sess,modpath='./modelres/Epoc0Iter20999.ckpt')
M.loadSess('./modelres/',sess=sess,modpath='./l2res/Epoc0Iter11999.ckpt')

def eval(img):
	res = sess.run(featurelayer,feed_dict={imgholder:img})
	return res

def __exit__():
	sess.close()