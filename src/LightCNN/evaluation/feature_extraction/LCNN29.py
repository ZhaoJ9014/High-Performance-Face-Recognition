import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import model as M
import numpy as np

def LCNN29():
	with tf.name_scope('img_holder'):
		img_holder = tf.placeholder(tf.float32, [None, 128, 128, 3])

	mod = M.Model(img_holder, [None, 128, 128, 3])

	mod.conv_layer(5, 96, activation=1)
	mod.maxpooling_layer(2, 2) #pool1

	# 
	a = mod.get_current_layer()
	mod.conv_layer(3, 96, activation=1)
	mod.conv_layer(3, 96, activation=1)
	# print (mod.get_shape())

	mod.sum(a)
	mod.conv_layer(1, 96, activation=1)
	mod.conv_layer(3, 192, activation=1)
	mod.maxpooling_layer(2, 2) #pool2

	# 
	b = mod.get_current_layer()
	mod.conv_layer(3, 96*2, activation=1)
	mod.conv_layer(3, 96*2, activation=1)

	mod.sum(b)
	mod.conv_layer(1, 96, activation=1)
	mod.conv_layer(3, 384, activation=1)
	mod.maxpooling_layer(2, 2) #pool3

	# 
	c = mod.get_current_layer()
	mod.conv_layer(3, 192*2, activation=1)
	mod.conv_layer(3, 192*2, activation=1)

	mod.sum(c)
	mod.conv_layer(1, 384, activation=1)
	mod.conv_layer(3, 256, activation=1)

	# 
	d = mod.get_current_layer()
	mod.conv_layer(3, 128*2, activation=1)
	mod.conv_layer(3, 128*2, activation=1)

	mod.sum(d)
	mod.conv_layer(1, 256, activation=1)
	mod.conv_layer(3, 256, activation=1)
	mod.maxpooling_layer(2, 2) #pool4

	mod.flatten()
	mod.fcnn_layer(512)
	feature_layer = mod.get_current_layer()[0]

	return feature_layer, img_holder

with tf.variable_scope('LCNN29'):
	feature_layer, img_holder = LCNN29()

# 
sess = tf.Session()
model_path = './model/Epoc_49_Iter_663.cpkt'
M.loadSess(model_path, sess)

def eval(img):
	res = sess.run(feature_layer, feed_dict={img_holder:img})
	return res

def __exit__():
	sess.close()
