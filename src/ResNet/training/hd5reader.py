import h5py
import numpy as np 
import cv2
import random
import time

###########################################################
#Read from hd5
#dynamic read data
class hd5reader():
	def __init__(self,dblist,valdb,bsize,vsize):
		self.bsize = bsize
		self.vsize = vsize
		f = open(dblist,'r')
		self.dbs = []
		for line in f:
			self.dbs.append(line.replace('\n',''))
		random.shuffle(self.dbs)
		print('initializing data reader...')
		print(self.dbs[0])
		f1 = h5py.File(self.dbs[0])
		print('listing data...')
		self.dt = list(np.float32(f1['data']))
		print('listing label...')
		self.lb = list(np.array(f1['label']))
		print(self.dt[0].shape)
		self.db_pos = 1
		f1.close()
		self.epoc = int(len(self.dbs)*len(self.dt)/self.bsize)
		print('reading val data...')
		f2 = h5py.File(valdb)
		self.val = list(np.float32(f2['data']))
		self.vallb = list(np.array(f2['label']))
		f2.close()

	def getEpoc(self):
		return self.epoc

	def readdb(self):
		print('reading database number',self.db_pos)
		fname = self.dbs[self.db_pos]
		self.db_pos+=1
		if self.db_pos==len(self.dbs):
			self.db_pos=0
			random.shuffle(self.dbs)
		a = time.time()
		f = h5py.File(fname)
		dt = list(np.float32(f['data']))
		lb = list(np.array(f['label']))
		self.dt.extend(dt)
		self.lb.extend(lb) 
		b = time.time()
		f.close()
		print('Time spent for data reading:',str(b-a))

	def checkmemory(self):
		print('\n\n\n\nstarting data check process...')
		print('check size:',len(self.dt))
		if len(self.dt)<50*self.bsize:
			self.readdb()

	def train_nextbatch(self,rd=False):
		dt = self.dt[:self.bsize]
		lb = self.lb[:self.bsize]
		self.dt = self.dt[self.bsize:]
		self.lb = self.lb[self.bsize:]
		w = random.randint(0,16)
		h = random.randint(0,16)
		if rd:
			dt = np.float32(dt)[:,w:w+128,h:h+128]
		else:
			dt = np.float32(dt)
		lb = np.array(lb).reshape([-1])
		print(lb[0])
		return dt,lb

	def val_nextbatch(self,rd=False):
		dt = self.val[:self.vsize]
		self.val[-self.vsize:],self.val[:-self.vsize] = self.val[:self.vsize],self.val[self.vsize:]
		lb = self.vallb[:self.vsize]
		self.vallb[-self.vsize:],self.vallb[:-self.vsize] = self.vallb[:self.vsize],self.vallb[self.vsize:]
		w = random.randint(0,16)
		h = random.randint(0,16)
		if rd:
			dt = np.float32(dt)[:,w:w+128,h:h+128]
		else:
			dt = np.float32(dt)
		lb = np.array(lb).reshape([-1])
		print(lb[0])
		return dt,lb