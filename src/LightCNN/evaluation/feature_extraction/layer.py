import tensorflow as tf
import numpy as np

def weight(shape):
	return tf.get_variable('weight', shape, initializer=tf.contrib.layers.xavier_initializer())

def bias(shape, value=0.1):
	return tf.get_variable('bias', shape, initializer=tf.constant_initializer(value))


def Fcnn(x,insize,outsize,name,activation=None,nobias=False):
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize])
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize])
			b = bias([outsize])
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def conv2D(x,kernel_size,outchn,name,stride=1,pad='SAME', usebias=True):
	print('Conv_bias:',usebias)
	# with tf.variable_scope(name):
	# if isinstance(size,list):
	# 	kernel = size
	# else:
	kernel = [kernel_size, kernel_size]
	z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
		kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
		use_bias=usebias,\
		bias_initializer=tf.constant_initializer(0.1),name=name)
	# print ('z:', z.get_shape())
	return z

def maxpooling(x,size,stride,name,pad='SAME'):
	with tf.variable_scope(name):
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def avgpooling(x,size,stride,name,pad='SAME'):
	with tf.variable_scope(name):
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)


def MFM(x,name):
	with tf.variable_scope(name):
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

def batch_norm(inp,name,training=True):
	print('BN training:',training)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def L2_norm(inp, dim):
	print ('L2 normlization...')
	return tf.nn.l2_norm(inp, dim)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sparse_softmax_cross_entropy(inp,lab,name):
	with tf.name_scope(name):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=inp))
		return loss

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)

def resize_nn(inp,size,name):
	with tf.name_scope(name):
		return tf.image.resize_nearest_neighbor(inp,size=(int(size),int(size)))

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		return acc

def dropout(inp, keep_prob):
	return tf.nn.dropout(inp,keep_prob)
