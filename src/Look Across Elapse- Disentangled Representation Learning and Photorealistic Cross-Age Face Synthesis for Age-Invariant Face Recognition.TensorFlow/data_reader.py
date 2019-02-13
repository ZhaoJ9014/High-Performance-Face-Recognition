import numpy as np 
import cv2 
import glob 
import random

AGE_CATEGORIES = 10

class data_reader():
	def __init__(self):
		self.load_data()
		random.shuffle(self.data)
		self.pos = 0

	def process_age(self, age):
		if 0 <= age <= 5:
			age = 0
		elif 6 <= age <= 10:
			age = 1
		elif 11 <= age <= 15:
			age = 2
		elif 16 <= age <= 20:
			age = 3
		elif 21 <= age <= 30:
			age = 4
		elif 31 <= age <= 40:
			age = 5
		elif 41 <= age <= 50:
			age = 6
		elif 51 <= age <= 60:
			age = 7
		elif 61 <= age <= 70:
			age = 8
		else:
			age = 9
		return age

	def load_data(self):
		print('Loading data...')
		data = []
		for i in glob.glob('./data/UTKFace/*.jpg'):
			img = cv2.imread(i)
			img = cv2.resize(img, (128, 128))
			i = i.replace('\\','/').split('/')[-1]
			i = i.split('_')
			age = int(i[0])
			gender = int(i[1])

			age = np.eye(AGE_CATEGORIES)[self.process_age(age)]
			gender = np.eye(2)[gender]
			data.append([img, age, gender])
		self.data = data
		print('Load finished.')

	def process_image(self,img):
		img = np.float32(img) / 127.5 - 1.
		return img 

	def get_next_batch(self, bsize):
		if self.pos + bsize > len(self.data):
			random.shuffle(self.data)
			self.pos = 0

		batch = self.data[self.pos: self.pos+bsize]
		self.pos += bsize

		img_batch, age_batch, gender_batch = list(zip(*batch))

		img_batch = self.process_image(img_batch)
		age_batch = np.float32(age_batch)
		gender_batch = np.float32(gender_batch)
		
		return img_batch, age_batch, gender_batch

