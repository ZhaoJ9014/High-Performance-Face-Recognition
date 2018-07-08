import h5py
import cv2
import random
import numpy as np

class hd5_reader():
	def __init__(self, train_list, val_list, train_bsize, val_bsize):
		print ('initial...')
		self.train_list = []
		self.val_list = []
		self.train_bsize = train_bsize
		self.val_bsize = val_bsize
		self.BSIZE = train_bsize
		self.train_data = []
		self.train_label = []
		self.val_data = []
		self.val_label = []
		self.train_data_iter = 0
		self.read_data_flag = 0
		# self.ITER_flag = True

		# train
		f = open(train_list)
		for line in f:
			line = line.strip()
			self.train_list.append(line)
		f.close()
		random.shuffle(self.train_list)
		print ('training list: ' + str(len(self.train_list)))

		f = h5py.File(self.train_list[0])
		self.train_data = np.float32(f['data'])
		self.train_label = np.int32(f['label'])
		f.close()
		print ('train data: ' + str(self.train_data.shape))
		print ('train data: ' + str(self.train_data[0].shape))

		self.train_data_ITERS = float(len(self.train_data)/float(self.train_bsize))
		if self.train_data_ITERS % 1 == 0:
			self.train_data_ITERS = int(self.train_data_ITERS)
		else:
			self.train_data_ITERS = int(self.train_data_ITERS) + 1
		print ('train data iters: ' + str(self.train_data_ITERS))

		self.train_epoc = int(len(self.train_data)/self.train_bsize)*len(self.train_list) + len(self.train_list)
		self.train_list_num = 1
		print ('train data epoc: ' + str(self.train_epoc))
		print ('train data list num: ' + str(self.train_list_num))

		# validation
		self.val_data_iter = 0
		f = open(val_list)
		for line in f:
			line = line.strip()
			self.val_list.append(line)
		f.close()
		print ('val list: ' + str(len(self.val_list)))

		for i in range(len(self.val_list)):
			f = h5py.File(self.val_list[i])
			self.val_data.extend(np.float32(f['data']))
			self.val_label.extend(np.int32(f['label']))
			f.close()
		print ('val data: ' + str(len(self.val_data)))
		print ('val data: ' + str(self.val_data[0].shape))

		self.val_data_ITERS = float(len(self.val_data)/float(self.val_bsize))
		if self.val_data_ITERS % 1 == 0:
			self.val_data_ITERS = int(self.val_data_ITERS)
		else:
			self.val_data_ITERS = int(self.val_data_ITERS) + 1
		print ('val data iters: ' + str(self.val_data_ITERS))
		# raw_input()

	def read_train_data(self):
		print ('read next hd5 file...')
		self.train_data = []
		self.train_label = []
		self.train_data_iter = 1
		self.train_bsize = self.BSIZE
		self.read_data_flag = 0
		# self.ITER_flag = True

		if self.train_list_num < len(self.train_list): 
			f = h5py.File(self.train_list[self.train_list_num])
			self.train_data = np.float32(f['data'])
			self.train_label = np.int32(f['label'])
			f.close()
			print ('train data: ' + str(self.train_data.shape))
			print ('train data: ' + str(self.train_data[0].shape))

			self.train_data_ITERS = float(len(self.train_data)/float(self.train_bsize))
			if self.train_data_ITERS % 1 == 0:
				self.train_data_ITERS = int(self.train_data_ITERS)
			else:
				self.train_data_ITERS = int(self.train_data_ITERS) + 1

			self.train_list_num += 1
		else:
			print ('shuffle data...')
			self.train_list_num = 0
			self.train_data = []
			self.train_label = []
			self.train_data_iter = 1
			self.train_bsize = self.BSIZE
			self.read_data_flag = 0
			# self.ITER_flag = False

			random.shuffle(self.train_list)
			f = h5py.File(self.train_list[self.train_list_num])
			self.train_data = np.float32(f['data'])
			self.train_label = np.int32(f['label'])
			f.close()
			print ('train data: ' + str(self.train_data.shape))
			print ('train data: ' + str(self.train_data[0].shape))

			self.train_data_ITERS = float(len(self.train_data)/float(self.train_bsize))
			if self.train_data_ITERS % 1 == 0:
				self.train_data_ITERS = int(self.train_data_ITERS)
			else:
				self.train_data_ITERS = int(self.train_data_ITERS) + 1
			print (self.train_data_ITERS)
			self.train_list_num += 1

	def train_nextbatch(self):
		self.train_data_iter += 1
		if self.read_data_flag == 1:
			self.read_train_data()
			
			train_data_batch = self.train_data[(self.train_data_iter-1)*self.train_bsize:self.train_data_iter*self.train_bsize]
			train_label_batch = self.train_label[(self.train_data_iter-1)*self.train_bsize:self.train_data_iter*self.train_bsize]

		elif self.train_data_iter == self.train_data_ITERS:
			train_data_batch = self.train_data[(self.train_data_iter-1)*self.train_bsize:]
			train_label_batch = self.train_label[(self.train_data_iter-1)*self.train_bsize:]
			self.train_bsize = len(train_data_batch)
			self.read_data_flag = 1
		else:
			train_data_batch = self.train_data[(self.train_data_iter-1)*self.train_bsize:self.train_data_iter*self.train_bsize]
			train_label_batch = self.train_label[(self.train_data_iter-1)*self.train_bsize:self.train_data_iter*self.train_bsize]

		w = random.randint(0, 16)
		h = random.randint(0, 16)
		train_data_batch = np.float32(train_data_batch)[:,w:w+128,h:h+128]
		train_data_batch = train_data_batch/255.
		train_label_batch = np.array(train_label_batch).reshape(-1)
		return train_data_batch, train_label_batch

	def val_nextbatch(self):
		self.val_data_iter += 1
		
		if self.val_data_iter == self.val_data_ITERS:
			val_data_batch = self.val_data[(self.val_data_iter-1)*self.val_bsize:]
			val_label_batch = self.val_label[(self.val_data_iter-1)*self.val_bsize:]
			self.val_bsize = len(val_data_batch)
		else:
			val_data_batch = self.val_data[(self.val_data_iter-1)*self.val_bsize:self.val_data_iter*self.val_bsize]
			val_label_batch = self.val_label[(self.val_data_iter-1)*self.val_bsize:self.val_data_iter*self.val_bsize]

		w = 8
		h = 8
		val_data_batch = np.float32(val_data_batch)[:,w:w+128,h:h+128]
		# val_data_batch = np.float32(val_data_batch)[:,8:136,8:136]
		val_data_batch = val_data_batch/255.
		val_label_batch = np.array(val_label_batch).reshape(-1)
		return val_data_batch, val_label_batch

















