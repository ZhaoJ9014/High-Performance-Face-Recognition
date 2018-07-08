import tensorflow as tf
import layer as L

class Model():
	def __init__(self, res, res_size):
		self.res = res
		self.res_size = res_size
		self.layer_num = 0
		self.batch_norm = True

	def get_current_layer_res(self):
		return self.res

	def get_shape(self):
		return self.res_size

	def get_current_layer(self):
		return [self.res, list(self.res_size)]

	def activate_layer(self, param):
		# res = self.res
		with tf.name_scope('activation_' + str(self.layer_num)):
			if param == 1:
				self.res = L.MFM(self.res, name='mfm_'+str(self.layer_num))
				self.res_size = self.res.get_shape()
			else:
				self.res = self.res
				self.res_size = self.res_size
		return [self.res, self.res_size]

	def conv_layer(self, kernel_size, outchn, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		with tf.name_scope('conv_' + str(self.layer_num)):
			inp_size = self.res_size
			self.res = L.conv2D(self.res, kernel_size, outchn, 'conv_'+str(self.layer_num), stride=stride, pad=pad, usebias=usebias)
			self.res_size = self.res.get_shape()

			if batch_norm:
				self.res = batch_norm(self.res, 'batch_norm_'+str(self.layer_num), training=self.batch_norm)

			self.activate_layer(activation)
			self.layer_num += 1

			print ('conv_'+str(self.layer_num), kernel_size, inp_size, self.res_size)
			return [self.res, self.res_size]

	def maxpooling_layer(self, kernel_size, stride=1, pad='SAME'):
		with tf.name_scope('maxpooling_' + str(self.layer_num)):
			inp_size = self.res_size
			self.res = L.maxpooling(self.res, kernel_size, stride=stride, name='maxpooling_'+str(self.layer_num), pad=pad)
			self.res_size = self.res.get_shape()

			print ('maxpooling_'+str(self.layer_num), kernel_size, inp_size, self.res_size)
			return [self.res, self.res_size]

	def sum(self, layerin):
		layer1_size = self.res_size
		layer2_size = layerin[1]
		print (layer1_size[2], layer1_size[3])
		print (layer2_size[2], layer2_size[3])
		assert layer1_size[1] == layer2_size[1]
		assert layer1_size[2] == layer2_size[2] and layer1_size[3] == layer2_size[3]

		inp_size = self.res_size
		with tf.name_scope('sum_' + str(self.layer_num)):
			self.res = self.res + layerin[0]
			self.res_size = self.res.get_shape()

		print ('sum_'+str(self.layer_num), inp_size, self.res_size)
		return [self.res, self.res_size]

	def flatten(self):
		inp_size = self.res_size
		# print (inp_size[1], inp_size[2], inp_size[3])
		with tf.name_scope('flatten_' + str(self.layer_num)):
			self.res = tf.reshape(self.res, [-1, int(inp_size[1])*int(inp_size[2])*int(inp_size[3])])
			self.res_size = self.res.get_shape()

		print ('flatten_'+str(self.layer_num), inp_size, self.res_size)
		return [self.res, self.res_size]

	def fcnn_layer(self, outsize, nobias=False):
		# inp_size = int(self.res.get_shape()[1])*int(self.res.get_shape()[2])*int(self.res.get_shape()[3])
		inp_size = self.res.get_shape()[1]
		# print ('inp_size', inp_size)
		# print ('outsize', outsize)
		with tf.name_scope('Fcnn_' + str(self.layer_num)):
			self.res = L.Fcnn(self.res, inp_size, outsize, 'Fcnn_'+str(self.layer_num), activation=None, nobias=False)
			self.res_size = self.res.get_shape()

		self.layer_num += 1
		print ('Fcnn_'+str(self.layer_num), inp_size, self.res_size)
		return [self.res, self.res_size]

	def dropout(self, keep_prob):
		self.res = L.dropout(self.res, keep_prob)
		self.res_size = self.res.get_shape()

		return [self.res, self.res_size]

	def accuracy(self, label):
		self.res = L.accuracy(self.res, label, 'accuracy')
		
		return self.res


def loadSess(model_path, sess):

	epoc = 0
	iters = 0
	if model_path != None:
		ckpt = tf.train.get_checkpoint_state(model_path)
		if ckpt:
			model = ckpt.model_checkpoint_path
			epoc = model.split('_')[1]
			iters = model.split('_')[-1].replace('.cpkt', '')
			print ('loading from model:', model)

			saver = tf.train.Saver()
			saver.restore(sess, model)
		else:
			sess.run(tf.global_variables_initializer())
			print ('No checkpoint in the folder...')

	return sess, int(epoc), int(iters)




