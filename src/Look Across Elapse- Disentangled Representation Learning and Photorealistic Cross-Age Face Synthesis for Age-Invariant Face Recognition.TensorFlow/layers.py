import tensorflow as tf 
import numpy as np 

l_num = 0

###########################################################
#define weight and bias initialization

def weight(shape,dtype=None):
	return tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)

def bias(shape,value=0.0,dtype=None):
	return tf.get_variable('bias',shape,initializer=tf.constant_initializer(value),dtype=dtype)

###########################################################
#define basic layers

def Fcnn(x,insize,outsize,name,activation=None,nobias=False,dtype=None):
	if dtype is None:
		dtype = x.dtype
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize],dtype=dtype)
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize],dtype=dtype)
			b = bias([outsize],dtype=dtype)
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def MFM(x,half,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		# shape = tf.shape(x)
		shape = x.get_shape().as_list()
		res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2])
		res = tf.reduce_max(res,axis=[3])
		return res

def MFMfc(x,half,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		# print('fcshape:',shape)
		res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),reduction_indices=[1])
	return res

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		#acc = tf.cast(correct,tf.float32)
		return acc

def batch_norm(inp,name,epsilon=None,variance=None,training=True):
	print('BN training:',training)
	if not epsilon is None:
		return tf.layers.batch_normalization(inp,training=training,name=name,epsilon=epsilon)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)
