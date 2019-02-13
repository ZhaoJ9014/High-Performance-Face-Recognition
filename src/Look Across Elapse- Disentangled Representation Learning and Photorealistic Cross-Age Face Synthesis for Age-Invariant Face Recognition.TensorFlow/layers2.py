import tensorflow as tf 
import numpy as np 

def set_gpu(config_str):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = config_str

###########################################################
#define weight and bias initialization
def weight(shape,data=None,dtype=None):
	if dtype is None:
		dtype = tf.float32
	if data is not None:
		w = tf.get_variable('weight',shape,initializer=tf.constant_initializer(data),dtype=dtype)
	else:
		w = tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)
	return w

def weight_conv(shape,data=None,dtype=None):
	if dtype is None:
		dtype = tf.float32
	if data is not None:
		k = tf.get_variable('kernel',shape,initializer=tf.constant_initializer(data),dtype=dtype)
	else:
		k = tf.get_variable('kernel',shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(),dtype=dtype)
	return k 

def bias(shape,name='bias',value=0.0,dtype=None,trainable=True):
	if dtype is None:
		dtype = tf.float32
	b = tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(value),dtype=dtype,trainable=trainable)
	return b

###########################################################
#define layer class
class Layer(tf.contrib.checkpoint.Checkpointable):
	def __init__(self, name):
		# template for layer definition
		self.initialized = False
		self.variables = []
		if not name is None:
			with tf.variable_scope(name):
				if not tf.executing_eagerly():
					self._parse_args()
					self._initialize()
					self.initialized = True
					self.output = self._deploy()
		else:
			if not tf.executing_eagerly():
				self._parse_args()
				self._initialize()
				self.initialized = True
				self.output = self._deploy()

	def _add_variable(self,var):
		# if not hasattr(self,'variables'):
		# 	self.variables = []
		self.variables.append(var)

	def _initialize(self):
		pass

	def _parse_args(self):
		pass

	def __call__(self, x):
		self.x = tf.convert_to_tensor(x)
		if not self.initialized:
			self._parse_args()
			self._initialize()
			self.initialized = True
		return self._deploy()

###########################################################
#define basic layers

class conv2D(Layer):
	def __init__(self,size,outchn,x=None,name=None,stride=1,pad='SAME',usebias=True,values=None,kernel_data=None,bias_data=None,dilation_rate=1,weight_norm=False):
		self.x = x
		self.size = size
		self.outchn = outchn
		self.name = name
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		if values is None:
			self.kernel_data = None
			self.bias_data = None
		else:
			self.kernel_data = values[0]
			self.bias_data = values[1]
		self.dilation_rate = dilation_rate
		self.weight_norm = weight_norm

		super().__init__(name)

	def _parse_args(self):
		# set size
		inchannel = self.x.get_shape().as_list()[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,1]

	def _initialize(self):
		# this will enlarge ckpt size. (at first time)
		if self.kernel_data is not None:
			self.W = weight_conv(self.kernel_data.shape, self.kernel_data)
		else:
			self.W = weight_conv(self.size)
			if self.weight_norm:
				print('Enable weight norm')
				self.W = self.W.initialized_value()
				self.W = tf.nn.l2_normalize(self.W, [0,1,2])
				print('Initialize weight norm')
				x_init = tf.nn.conv2d(self.x,self.W,stride,pad,dilations=dilation_rate)
				m_init, v_init = tf.nn.moments(x_init,[0,1,2])
				s_init = 1. / tf.sqrt(v_init + 1e-8)
				s = tf.get_variable('weight_scale',dtype=tf.float32,initializer=s_init)
				self.S = s.initialized_value()
				self.S = tf.reshape(self.S,[1,1,1,outchn])
				self.W = self.S *self.W
				self._add_variable(self.S)
		self._add_variable(self.W)

		# 
		if self.usebias:
			if self.bias_data is not None:
				self.b = bias([self.outchn], value=self.bias_data)
			else:
				self.b = bias([self.outchn])
		self._add_variable(self.b)
		
	def _deploy(self):
		out = tf.nn.conv2d(self.x,self.W,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.b)
		return out 

class conv1D(Layer):
	def __init__(self,size,outchn,x=None,name=None,stride=1,pad='SAME',usebias=True,values=None,kernel_data=None,bias_data=None,dilation_rate=1,weight_norm=False):
		self.x = x
		self.size = size
		self.outchn = outchn
		self.name = name
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		if values is None:
			self.kernel_data = None
			self.bias_data = None
		else:
			self.kernel_data = values[0]
			self.bias_data = values[1]
		self.dilation_rate = dilation_rate
		self.weight_norm = weight_norm

		super().__init__(name)

	def _parse_args(self):
		# set size
		inchannel = self.x.get_shape().as_list()[-1]
		self.size = [1, self.size, inchannel, self.outchn]
		# set stride
		self.stride = [1,1, self.stride, 1]
		# set dilation
		self.dilation_rate = [1,1,self.dilation_rate,1]

	def _initialize(self):
		# this will enlarge ckpt size. (at first time)
		if self.kernel_data is not None:
			self.W = weight_conv(self.kernel_data.shape, self.kernel_data)
		else:
			self.W = weight_conv(self.size)
			if self.weight_norm:
				print('Enable weight norm')
				self.W = self.W.initialized_value()
				self.W = tf.nn.l2_normalize(self.W, [0,1,2])
				print('Initialize weight norm')
				x_init = tf.nn.conv2d(self.x,self.W,stride,pad,dilations=dilation_rate)
				m_init, v_init = tf.nn.moments(x_init,[0,1,2])
				s_init = 1. / tf.sqrt(v_init + 1e-8)
				s = tf.get_variable('weight_scale',dtype=tf.float32,initializer=s_init)
				self.S = s.initialized_value()
				self.S = tf.reshape(self.S,[1,1,1,outchn])
				self.W = self.S *self.W
				self._add_variable(self.S)
		self._add_variable(self.W)

		# 
		if self.usebias:
			if self.bias_data is not None:
				self.b = bias([self.outchn], value=self.bias_data)
			else:
				self.b = bias([self.outchn])
		self._add_variable(self.b)
		
	def _deploy(self):
		self.x = tf.expand_dims(self.x, axis=1)
		out = tf.nn.conv2d(self.x,self.W,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.b)
		out = tf.squeeze(out, axis=1)
		return out 

class maxpoolLayer(Layer):
	def __init__(self,size,x=None,stride=None,name=None,pad='SAME'):
		self.x = x 
		self.name = name
		self.size = size
		self.stride = stride
		self.pad = pad

		super().__init__(name)

	def _parse_args(self):
		if isinstance(self.size, list):
			if len(self.size)==2:
				self.size = [1, self.size[0], self.size[1], 1]
		elif isinstance(self.size, int):
			self.size = [1, self.size, self.size, 1]

		if not self.stride:
			self.stride = self.size
		elif isinstance(self.stride, list):
			if len(self.stride)==2:
				self.stride = [1,self.stride[0],self.stride[1],1]
		elif isinstance(self.stride, int):
			self.stride = [1, self.stride, self.stride, 1]

	def _deploy(self):
		return tf.nn.max_pool(self.x, ksize=self.size, strides=self.stride, padding=self.pad)

class activation(Layer):
	def __init__(self, param, x=None, name=None, **kwarg):
		self.x = x 
		self.param = param
		self.name = name
		self.kwarg = kwarg

		super().__init__(name)

	def _deploy(self):
		if self.param == 0:
			res =  tf.nn.relu(self.x)
		elif self.param == 1:
			if 'leaky' in self.kwarg:
				leaky = self.kwarg['leaky']
			else:
				leaky = 0.2
			res =  tf.maximum(self.x,self.x*leaky)
		elif self.param == 2:
			res =  tf.nn.elu(self.x)
		elif self.param == 3:
			res =  tf.tanh(self.x)
		elif self.param == 4:
			shape = self.x.get_shape().as_list()
			res = tf.reshape(self.x,[-1,shape[1],shape[2],2,shape[-1]//2]) # potential bug in conv_net
			res = tf.reduce_max(res,axis=[3])
		elif self.param == 5:
			shape = self.x.get_shape().as_list()
			res = tf.reduce_max(tf.reshape(self.x,[-1,2,shape[-1]//2]),axis=[1])
		elif self.param == 6:
			res =  tf.sigmoid(self.x)
		else:
			res =  self.x
		return res

class fcLayer(Layer):
	def __init__(self, outsize, usebias=True, x=None, values=None, name=None):
		self.x = x 
		self.outsize = outsize
		self.usebias = usebias
		self.name = name 
		self.values = values

		super().__init__(name)

	def _initialize(self):
		insize = self.x.get_shape().as_list()[-1]
		if self.values is not None:
			self.W = weight([insize, self.outsize], data=self.values[0])
		else:
			self.W = weight([insize, self.outsize])
		self._add_variable(self.W)

		if self.usebias:
			if self.values is not None:
				self.b = bias([self.outsize], value=self.values[1])
			else:
				self.b = bias([self.outsize])
			self._add_variable(self.b)

	def _deploy(self):
		res = tf.matmul(self.x, self.W)
		if self.usebias:
			res = tf.nn.bias_add(res, self.b)
		return res 

class batch_norm_graph(Layer):
	def __init__(self, training, epsilon, x=None, name=None):
		assert (not tf.executing_eagerly()),'batch_norm_graph can only run in graph mode'
		self.x = x 
		self.training = training
		self.epsilon = epsilon
		self.name = name

		super().__init__(name)

	def _deploy(self):
		# will modify this to lower api in later version
		if not self.epsilon is None:
			return tf.layers.batch_normalization(self.x,training=self.training,name=self.name,epsilon=self.epsilon)
		return tf.layers.batch_normalization(self.x,training=self.training,name=self.name)

class batch_norm(Layer):
	def __init__(self, decay=0.01, epsilon=0.001, is_training=True, name=None, values=None):
		assert tf.executing_eagerly(),'batch_norm can only run in graph mode'
		self.name = name
		self.decay = decay
		self.epsilon = epsilon
		self.is_training = is_training
		self.values = values

		super().__init__(name)

	def _initialize(self):
		shape = self.x.get_shape().as_list()[-1]
		if self.values is None:
			self.moving_average = bias([shape],name='moving_average',value=0.0,trainable=False)
			self.variance = bias([shape],name='variance',value=1.0,trainable=False)

			self.gamma = bias([shape],name='gamma',value=1.0,trainable=True)
			self.beta = bias([shape],name='beta',value=0.0,trainable=True)
		else:
			self.moving_average = bias([shape],name='moving_average',value=self.values[0],trainable=False)
			self.variance = bias([shape],name='variance',value=self.values[1],trainable=False)

			self.gamma = bias([shape],name='gamma',value=self.values[2],trainable=True)
			self.beta = bias([shape],name='beta',value=self.values[3],trainable=True)

	def update(self,variable,value):
		delta = (variable - value) * self.decay
		variable.assign_sub(delta)

	def _deploy(self):
		inp_dim_num = len(self.x.get_shape().as_list())
		if inp_dim_num==3:
			self.x = tf.expand_dims(self.x, axis=1)
		if self.is_training:
			res, mean, var = tf.nn.fused_batch_norm(self.x, self.gamma, self.beta, None, None, self.epsilon, is_training=self.is_training)
			self.update(self.moving_average, mean)
			self.update(self.variance, var)
		else:
			res, mean, var = tf.nn.fused_batch_norm(self.x, self.gamma, self.beta, self.moving_average, self.variance, self.epsilon, is_training=self.is_training)
		if inp_dim_num==3:
			res = tf.squeeze(res , axis=1)
		return res 

class deconv2D(Layer):
	def __init__(self,size,outchn,x=None,stride=1,usebias=True,pad='SAME',name=None):
		self.x = x
		self.size = size 
		self.outchn = outchn
		self.name = name 
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias

		super().__init__(name)

	def _parse_args(self):
		inp_size = self.x.get_shape().as_list()
		inchannel = inp_size[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],self.outchn,inchannel]
		else:
			self.size = [self.size, self.size, self.outchn, inchannel]

		if isinstance(self.stride, list):
			if len(self.stride)==2:
				self.stride = [1,self.stride[0],self.stride[1],1]
		elif isinstance(self.stride, int):
			self.stride = [1, self.stride, self.stride, 1]

		# infer the output shape
		if self.pad == 'SAME':
			self.output_shape = [tf.shape(self.x)[0], tf.shape(self.x)[1]*self.stride[1], tf.shape(self.x)[2]*self.stride[2], self.outchn]
		else:
			self.output_shape = [tf.shape(self.x)[0], tf.shape(self.x)[1]*self.stride[1]+self.size[0]-self.stride[1], tf.shape(self.x)[2]*self.stride[2]+self.size[1]-self.stride[2], self.outchn]

	def _initialize(self):
		self.W = weight_conv(self.size)
		self._add_variable(self.W)
		if self.usebias:
			self.b = bias([self.outchn])
			self._add_variable(self.b)

	def _deploy(self):
		res = tf.nn.conv2d_transpose(self.x, self.W, self.output_shape, self.stride, padding=self.pad)
		if self.usebias:
			res = tf.nn.bias_add(res, self.b)
		return res 

class flatten(Layer):
	def __init__(self, x=None, name=None):
		self.x = x 

		super().__init__(name)

	def _deploy(self):
		shape = self.x.get_shape().as_list()
		num = 1
		for k in shape[1:]:
			num *= k
		res = tf.reshape(self.x, [-1, num])
		return res 


####### Functional layer #######
@tf.custom_gradient
def gradient_reverse(x):
	def grad(dy):
		return -dy
	return x, grad
