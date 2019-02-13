import tensorflow as tf 
import numpy as np 
import data_reader 
import modeleag as M 
import network
import cv2 
import os 

MAXITER = 100000

class AIM(M.Model):
	def initialize(self):
		self.encoder = network.EncoderNet()
		self.decoder = network.DecoderNet()
		self.dis_z = network.DiscriminatorZ()
		self.age_classifier = network.AgeClassifier() 
		self.dis_img = network.DiscriminatorPatch()

	def generate(self, x, age_batch, gender_batch, img):
		res, res_att, _ = self.decoder(self.encoder(x), age_batch, gender_batch, img)
		# can choose either res or res_att as output
		res = (res_att.numpy() + 1) * 127.5
		res = np.uint8(res)
		return res 

if __name__=='__main__':
	AIM_model = AIM()

	optim = tf.train.AdamOptimizer(0.0001)
	saver = M.Saver(AIM_model, optim)
	saver.restore('./model/')

	reader = data_reader.data_reader()

	# create result folder
	if not os.path.exists('./results/'):
		os.mkdir('./results/')

	# start training
	eta = M.ETA(MAXITER+1)
	for i in range(MAXITER+1):
		img_batch, age_batch, gender_batch = reader.get_next_batch(100)

		losses, loss_grad, tape = network.lossFunc(img_batch, age_batch, gender_batch, AIM_model)
		network.applyGrad(loss_grad, AIM_model, optim, tape)

		if i%10==0:
			network.printLosses(losses, i, eta)

		if i%1000==0:
			# visualize every 1000 iters
			for k in range(10):
				age_batch = np.zeros([age_batch.shape[0], 10],np.float32,)
				age_batch[:,k] = 1
				res = AIM_model.generate(img_batch, age_batch, gender_batch, img_batch)
				print(res.max())
				print(res.min())
				img_r = np.uint8((img_batch+1.)*127.5)
				for j in range(len(res)):
					cv2.imwrite('./results/%d_%d_r.jpg'%(i,j), img_r[j])
				for j in range(len(res)):
					cv2.imwrite('./results/%d_%d_%d.jpg'%(i,j,k), res[j])

		if i%2000==0 and i>0:
			saver.save('./model/%d.ckpt'%(i))
