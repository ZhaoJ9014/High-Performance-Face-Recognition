import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf 
import numpy as np 
import model as M 
from hd5reader import hd5reader
import cv2

CLASS = 20000
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
		imgholder = tf.placeholder(tf.float32,[None,128,128,3])
		img2 = tf.image.resize_images(imgholder,(224,224))
	with tf.name_scope('labholder'):
		labholder = tf.placeholder(tf.int64,[None,CLASS])
		print(labholder)
	mod = M.Model(img2,[None,224,224,3])
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
		classlayer,evallayer = M.enforcedClassfier(featurelayer,2048,labholder,BSIZE,CLASS,dropout=0.15,enforced=True)
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labholder,logits=classlayer))
		tf.summary.scalar('loss',loss)
	return classlayer,loss,imgholder,labholder,featurelayer,evallayer

with tf.variable_scope('MainModel'):
	classlayer,loss,imgholder,labholder,featurelayer,evallayer = res_18()

with tf.name_scope('accuracy'):
	acc = M.accuracy(evallayer,tf.argmax(labholder,1))
	tf.summary.scalar('accuracy',acc)

with tf.name_scope('optimizer'):
	extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	train = tf.train.AdamOptimizer(0.0001).minimize(loss)

modelpath = './resnet/'
logpath = './log/'
listFile = 'list.txt'
validationDB = '/home/psl/7T/ms_clean_0/train1_235.hd5'

def training(EPOC):
	with tf.Session() as sess:
		writer = tf.summary.FileWriter(logpath,sess.graph)
		merge = tf.summary.merge_all()
		M.loadSess(modelpath,sess=sess)
		reader = hd5reader(listFile,validationDB,BSIZE,BSIZE)
		saver = tf.train.Saver()
		ITERS = reader.getEpoc()
		counter = 0
		for i in range(EPOC):
			for x in range(ITERS):
				x_train,y_train = reader.train_nextbatch(rd=True)
				buff = np.zeros([BSIZE,CLASS],dtype=np.int64)
				for index in range(BSIZE):
					buff[index][y_train[index]] = 1
				y_train = buff
				# print(y_train.shape)
				_,_,ls1,ac,cls= sess.run([extra_update_ops,train,loss,acc,classlayer],feed_dict={imgholder:x_train,labholder:y_train})
				#print('clsmax',np.amax(cls,axis=1))
				#print(cls[0])
				#input()
				print('Epoc:',i,' |Iter:',x,' |Loss1:',ls1,' |Acc:',ac)
				if counter%10==0:
					reader.checkmemory()
				if counter%20==0:
					mg = sess.run(merge,feed_dict={imgholder:x_train,labholder:y_train})
					writer.add_summary(mg,counter)
				if x%200==0:
					lstotal = 0
					actotal = 0
					for j in range(20):
						d_val,l_val = reader.val_nextbatch(rd=True)
						buff = np.zeros([BSIZE,CLASS],dtype=np.int64)
						for index in range(BSIZE):
							buff[index][l_val[index]] = 1
						l_val = buff
						ls1,ac = sess.run([loss,acc],feed_dict={imgholder:d_val,labholder:l_val})
						lstotal += ls1
						actotal += ac 
					lstotal = lstotal/20
					actotal = actotal/20
					st = 'epoc:'+str(i)+'|val loss:'+str(lstotal)+' |acc:'+str(actotal)
					print (st)
				if (counter+1)%1000==0:
					saver.save(sess,modelpath+'Epoc'+str(i)+'Iter'+str(x)+'.ckpt')
				counter +=1

training(100)