import layers as L 
import tensorflow as tf
import numpy as np 
import os 

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

def set_gpu(config_str):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = config_str

class Model():
	def __init__(self,inp,size=None):
		self.result = inp
		if size is None:
			self.inpsize = inp.get_shape().as_list()
		else:
			self.inpsize = list(size)
		self.layernum = 0
		self.bntraining = True
		self.epsilon = None

	def set_bn_training(self,training):
		self.bntraining = training

	def set_bn_epsilon(self,epsilon):
		self.epsilon = epsilon

	def get_current(self):
		return self.get_current_layer()

	def get_current_layer(self):
		return self.result

	def __call__(self):
		return [self.result,self.inpsize]

	def get_shape(self):
		return self.inpsize

	def activation(self,param):
		return self.activate(param)

	def activate(self,param):
		inp = self.result
		with tf.name_scope('activation_'+str(self.layernum)):
			if param == 0:
				res =  L.relu(inp,name='relu_'+str(self.layernum))
			elif param == 1:
				res =  L.lrelu(inp,name='lrelu_'+str(self.layernum))
			elif param == 2:
				res =  L.elu(inp,name='elu_'+str(self.layernum))
			elif param == 3:
				res =  L.tanh(inp,name='tanh_'+str(self.layernum))
			elif param == 4:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFM(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 5:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFMfc(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 6:
				res =  L.sigmoid(inp,name='sigmoid_'+str(self.layernum))
			else:
				res =  inp
		self.result = res
		return self.result

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		with tf.variable_scope('fc_'+str(self.layernum)):
			self.inpsize = [i for i in self.inpsize]
			self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.inpsize[1] = outsize
			self.activate(activation)
			self.layernum+=1
		return self.result

	def set_current(self,layerinfo):
		if isinstance(layerinfo,list):
			self.result = layerinfo[0]
			self.inpsize = layerinfo[1]
		else:
			self.result = layerinfo
			self.inpsize = self.result.get_shape().as_list()

	def set_current_layer(self,layerinfo):
		self.set_current(layerinfo)

	def dropout(self,ratio):
		with tf.name_scope('dropout'+str(self.layernum)):
			self.result = tf.nn.dropout(self.result,ratio)
		return self.result

	def gradient_flip_layer(self):
		with tf.variable_scope('Gradient_flip_'+str(self.layernum)):
			@tf.RegisterGradient("GradFlip")
			def _flip_grad(op,grad):
				return [tf.negative(grad)]

			g = tf.get_default_graph()
			with g.gradient_override_map({'Identity':'GradFlip'}):
				self.result = tf.identity(self.result)
		return self.result
