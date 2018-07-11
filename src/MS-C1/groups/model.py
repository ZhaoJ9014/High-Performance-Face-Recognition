import layers as L 
import tensorflow as tf

crsentpy = -1
acc = -1

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

def loadSess(modelpath,sess=None,modpath=None,mods=None,var_list=None):
#load session if there exist any models, and initialize the sess if not
	assert modpath==None or mods==None
	if sess==None:
		sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	if var_list==None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list)
	ckpt = tf.train.get_checkpoint_state(modelpath)
	if modpath!=None:
		mod = modpath
		print('loading from model:',mod)
		saver.restore(sess,mod)
	elif mods!=None:
		for m in mods:
			print('loading from model:',m)
			saver.restore(sess,m)
	elif ckpt:
		mod = ckpt.model_checkpoint_path
		print('loading from model:',mod)
		saver.restore(sess,mod)
	else:
		print('No checkpoint in folder, use initial graph...')
	return sess

def sparse_softmax_cross_entropy(inp,lab):
	global crsentpy
	crsentpy+=1
	return L.sparse_softmax_cross_entropy(inp,lab,'cross_entropy_'+str(crsentpy))

def accuracy(inp,lab):
	global acc
	acc +=1
	return L.accuracy(inp,lab,'accuracy_'+str(acc))

def enforcedClassfier(featurelayer,inputdim,lbholder,BSIZE,CLASS,enforced=False,dropout=1):
	featurelayer = tf.nn.dropout(featurelayer,dropout)
	w = L.weight([inputdim,CLASS])
	nfl = tf.nn.l2_normalize(featurelayer,1)
	buff = tf.matmul(nfl,tf.nn.l2_normalize(w,0))
	evallayer = tf.matmul(featurelayer,w)
	if enforced:
		floatlb = tf.cast(lbholder,tf.float32)
		lbc = tf.ones([BSIZE,CLASS],dtype=tf.float32) - floatlb
		cosmtx = tf.multiply(floatlb,buff)
		filteredmtx = tf.multiply(lbc,buff)
		cosmtx2 = (tf.minimum(cosmtx*0.9,cosmtx*1.))*floatlb
		#cosmtx2 = tf.multiply(cosmtx,floatlb)
		lstlayer = cosmtx2+filteredmtx
		# lstlayer = tf.matmul(featurelayer,w)*lstlayer
		nb = tf.norm(w,axis=0,keep_dims=True)
		nf = tf.norm(featurelayer,axis=1,keep_dims=True)
		lstlayer = nb*lstlayer
		lstlayer = nf*lstlayer
	else:
		lstlayer = evallayer
	return lstlayer,evallayer

class Model():
	def __init__(self,inp,size):
		self.result = inp
		self.inpsize = list(size)
		self.layernum = 0
		self.transShape = None
		self.varlist = []
		self.fcs = []
		self.bntraining = True

	def set_bn_training(self,training):
		self.bntraining = training

	def get_current_layer(self):
		return self.result

	def get_shape(self):
		return self.inpsize

	def get_current(self):
		return [self.result,list(self.inpsize)]

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
				# print('mfmsize',self.inpsize)
				res =  L.MFM(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 5:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFMfc(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 6:
				res =  L.sigmoid(inp,name='sigmoid_'+str(self.layernum))
			else:
				res =  inp
		self.result = res
		return [self.result,list(self.inpsize)]

	def convLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False,layerin=None):
		with tf.variable_scope('conv_'+str(self.layernum)):
			if isinstance(size,list):
				kernel = size
			else:
				kernel = [size,size]
			if layerin!=None:
				self.result=layerin[0]
				self.inpsize=list(layerin[1])
			self.result = L.conv2D(self.result,kernel,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad)
			self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining)
			self.layernum += 1
			if pad=='VALID':
				self.inpsize[1] -= kernel[0]-stride
				self.inpsize[2] -= kernel[1]-stride
			self.inpsize[1] = self.inpsize[1]//stride
			self.inpsize[2] = self.inpsize[2]//stride
			self.inpsize[3] = outchn
			self.activate(activation)
		return [self.result,list(self.inpsize)]

	def deconvLayer(self,kernel,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.deconv2D(self.result,kernel,outchn,'deconv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.layernum+=1
		self.inpsize[1] *= stride
		self.inpsize[2] *= stride
		self.inpsize[3] = outchn
		self.activate(activation)
		return [self.result,list(self.inpsize)]

	def maxpoolLayer(self,size,pad='SAME',stride=None):
		if stride==None:
			stride = size
		self.result = L.maxpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		if pad=='VALID':
			self.inpsize[1] -= size-stride
			self.inpsize[2] -= size-stride
		else:
			if self.inpsize[1]%2==1:
				self.inpsize[1]+=1
			if self.inpsize[2]%2==1:
				self.inpsize[2]+=1
		self.inpsize[1] = self.inpsize[1]//stride
		self.inpsize[2] = self.inpsize[2]//stride
		return [self.result,list(self.inpsize)]

	def avgpoolLayer(self,size,pad='SAME',stride=None):
		if stride==None:
			stride = size
		self.result = L.avgpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		if pad=='VALID':
			self.inpsize[1] -= size-stride
			self.inpsize[2] -= size-stride
		self.inpsize[1] = self.inpsize[1]//stride
		self.inpsize[2] = self.inpsize[2]//stride
		return [self.result,list(self.inpsize)]

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.transShape = [self.inpsize[1],self.inpsize[2],self.inpsize[3],0]
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		self.fcs.append(len(self.varlist))
		return [self.result,list(self.inpsize)]

	def construct(self,shape):
		self.result = tf.reshape(self.result,[-1,shape[0],shape[1],shape[2]])
		self.inpsize = [None,shape[0],shape[1],shape[2]]
		return [self.result,list(self.inpsize)]

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		with tf.variable_scope('fc_'+str(self.layernum)):
			self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
			if len(self.fcs)!=0:
				if self.fcs[-1] == len(self.varlist):
					self.transShape[-1] = outsize
			self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
			self.inpsize[1] = outsize
			self.activate(activation)
			self.layernum+=1
		return [self.result,list(self.inpsize)]

	def scale(self,number):
		with tf.variable_scope('scale_'+str(self.layernum)):
			self.result = self.result * number
		return [self.result,list(self.inpsize)]

	def sum(self,layerin):
		assert layerin[1][2] == self.inpsize[2] and layerin[1][1] == self.inpsize[1]
		assert layerin[1][3] == self.inpsize[3]
		with tf.variable_scope('sum_'+str(self.layernum)):
			self.result = self.result +	layerin[0]
		return [self.result,list(self.inpsize)]

	def NIN(self,size,outchn1,outchn2,activation=-1,batch_norm=False,pad='SAME'):
		with tf.variable_scope('NIN_'+str(self.layernum)):
			self.convLayer(1,outchn1,activation=activation,batch_norm=batch_norm)
			self.convLayer(size,outchn2,activation=activation,batch_norm=batch_norm,pad=pad)
		return [self.result,list(self.inpsize)]

	def incep(self,outchn1,outchn2,outchn3,outchn4,outchn5,activation=-1,batch_norm=False):
		with tf.variable_scope('Incep_'+str(self.layernum)):
			orignres = self.result
			orignsize = self.inpsize
			a,_ = self.NIN(3,outchn1,outchn2,activation=activation,batch_norm=batch_norm)
			asize = self.inpsize
			self.inpsize = orignsize
			self.result = orignres
			b,_ = self.NIN(5,outchn3,outchn4,activation=activation,batch_norm=batch_norm)
			bsize = self.inpsize
			self.inpsize = orignsize
			self.result = orignres
			c,_ = self.convLayer(1,outchn5,activation=activation,batch_norm=batch_norm)
			csize = self.inpsize
			self.inpsize[3] = asize[3]+bsize[3]+csize[3]
			self.result = tf.concat(axis=3,values=[a,b,c])
			return [self.result,list(self.inpsize)]

	def concat_to_current(self,layerinfo):
		with tf.variable_scope('concat'+str(self.layernum)):
			layerin, layersize = layerinfo[0],list(layerinfo[1])
			assert layersize[2] == self.inpsize[2] and layersize[1]==self.inpsize[1]
			self.result = tf.concat(axis=3,values=[self.result,layerin])
			self.inpsize[3] += layersize[3]
		return [self.result,list(self.inpsize)]

	def set_current_layer(self,layerinfo):
		layerin, layersize = layerinfo[0],layerinfo[1]
		self.result = layerin
		self.inpsize = layersize

	def dropout(self,ratio):
		with tf.name_scope('dropout'+str(self.layernum)):
			self.result = tf.nn.dropout(self.result,ratio)
		return [self.result,list(self.inpsize)]

	def l2norm(self):
		with tf.name_scope('l2norm'+str(self.layernum)):
			self.result = tf.nn.l2_normalize(self.result,1)
		return [self.result,list(self.inpsize)]

	def batch_norm(self):
		with tf.variable_scope('batch_norm'+str(self.layernum)):
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		return [self.result,list(self.inpsize)]

	def convertVariablesToCaffe(self,sess,h5name):
		import caffeconverter as cc
		import scipy.io as sio 
		print('varlist:',len(self.varlist))
		f = open('layers.txt')
		dt = {}
		layers = []
		for line in f:
			layers.append(line.replace('\n',''))
		f.close()
		print('layers:',len(layers))
		print('variables:',len(self.varlist))
		for i in range(len(layers)):
			if i*2 in self.fcs:
				print('reshape fc layer...')
				dt[layers[i]+'w'] = cc.reshapeFcWeight(self.varlist[i*2],self.transShape,sess)
			else:
				dt[layers[i]+'w'] = sess.run(self.varlist[i*2])
			dt[layers[i]+'b'] = sess.run(self.varlist[i*2+1])
		sio.savemat('tfModelVars.mat',dt)
		cvt = cc.h5converter(h5name)
		cvt.startConvert()