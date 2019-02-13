import tensorflow as tf 
import numpy as np 
import layers2 as L 
L.set_gpu('0')
import modeleag as M 

IMG_SIZE = 128
AGE_CATEGORY = 10
Z_DIM = 50

class EncoderNet(M.Model):
	def initialize(self):
		# should modify the network structure for better training
		self.c1 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(5, 256, stride=2, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(5, 512, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(5, 1024, stride=2, activation=M.PARAM_RELU)
		self.fc = M.Dense(Z_DIM)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.fc(M.flatten(x))
		return tf.nn.tanh(x) 

class DecoderNet(M.Model):
	def initialize(self):
		# should modify the network structure for better training
		self.fc = M.Dense(4*4*1024, activation=M.PARAM_RELU)
		self.c1 = M.DeconvLayer(5, 512, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.DeconvLayer(5, 256, stride=2, activation=M.PARAM_RELU)
		self.c3 = M.DeconvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c4 = M.DeconvLayer(5, 64, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.DeconvLayer(5, 32, stride=2, activation=M.PARAM_RELU)
		self.c6 = M.DeconvLayer(5, 16, stride=1, activation=M.PARAM_RELU)
		self.c7 = M.DeconvLayer(5, 3, stride=1)

		# attention
		self.a1 = M.DeconvLayer(5, 32, stride=2, activation=M.PARAM_RELU)
		self.a2 = M.DeconvLayer(5, 16, stride=1, activation=M.PARAM_RELU)
		self.a3 = M.DeconvLayer(5, 1)

	def forward(self, x, age, gender, img):
		age = tf.tile(age, [1, 10])
		gender = tf.tile(gender, [1, 25])
		x = tf.concat([x,age,gender], axis=-1)
		x = self.fc(x)
		x = tf.reshape(x,[-1,4,4,1024])
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		att = x
		x = self.c5(x)
		x = self.c6(x)
		x = self.c7(x)
		x = tf.nn.tanh(x)

		att = self.a1(att)
		att = self.a2(att)
		att = self.a3(att) 
		att = tf.sigmoid(att)
		return x, img * att + x * (1.-att), att

class DiscriminatorZ(M.Model):
	def initialize(self):
		self.fc1 = M.Dense(128, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(32, activation=M.PARAM_RELU)
		self.fc3 = M.Dense(1)

	def forward(self, x):
		return self.fc3(self.fc2(self.fc1(x)))

class DiscriminatorPatch(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(5, 32, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(5,128, stride=2, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(5,256, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(1, 1) # discrimination score

		self.c6 = M.ConvLayer(5, 512, stride=2, activation=M.PARAM_RELU)
		self.fc = M.Dense(512, activation=M.PARAM_RELU)
		self.fc_age = M.Dense(AGE_CATEGORY)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		discrminate = self.c5(x)

		x = self.c6(x)
		x = M.flatten(x)
		x = self.fc(x)
		age = self.fc_age(x)
		return discrminate, age

class AgeClassifier(M.Model):
	def initialize(self):
		self.fc1 = M.Dense(128, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(64, activation=M.PARAM_RELU)
		self.fc3 = M.Dense(AGE_CATEGORY)

	def forward(self, x, reverse_grad):
		if reverse_grad:
			x = L.gradient_reverse(x)
		x = self.fc3(self.fc2(self.fc1(x)))
		return x 

def disLoss(d_real, d_fake):
	# use Mean Square Gan loss
	d_loss_real = tf.reduce_mean(tf.square(d_real - tf.ones_like(d_real)))
	d_loss_fake = tf.reduce_mean(tf.square(d_fake - tf.zeros_like(d_fake)))
	d_loss = (d_loss_real + d_loss_fake) * 0.5

	g_loss = tf.reduce_mean(tf.square(d_fake - tf.ones_like(d_fake)))
	return d_loss, g_loss

def ageLoss(pred, label):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label))
	return loss 

def tvLoss(x):
	loss = tf.reduce_mean(tf.image.total_variation(x))
	return loss 

def mseLoss(pred, label):
	loss = tf.reduce_mean(tf.abs(pred - label))
	return loss 

def lossFunc(img, label_age, label_gender, AIM):
	with tf.GradientTape() as tape:
		z_enc = AIM.encoder(img)
		img_fake, img_fake_att, att = AIM.decoder(z_enc, label_age, label_gender, img)

		# feature discriminator
		dis_z_fake = AIM.dis_z(z_enc)
		dis_z_real = AIM.dis_z(tf.random.uniform(z_enc.shape, -1., 1.))

		# age classifier
		age_pred = AIM.age_classifier(z_enc, reverse_grad=False)
		age_pred_r = AIM.age_classifier(z_enc, reverse_grad=True)

		# image discriminator
		dis_img_fake, age_fake = AIM.dis_img(img_fake)
		dis_img_fake_att, age_fake_att = AIM.dis_img(img_fake_att)
		dis_img_real, age_real = AIM.dis_img(img)

		# build losses 
		# reconstruction loss
		loss_img = mseLoss(img, img_fake) + 0.00*tvLoss(img_fake) + 0.001*tvLoss(att) + 0.01*tf.reduce_mean(tf.square(att))
		loss_dis_z_d, loss_dis_z_g = disLoss(dis_z_real, dis_z_fake)
		loss_dis_img_d, loss_dis_img_g1 = disLoss(dis_img_real, dis_img_fake)
		loss_dis_img_d, loss_dis_img_g2 = disLoss(dis_img_real, dis_img_fake_att)
		loss_dis_img_g = loss_dis_img_g1 + loss_dis_img_g2

		# c loss
		loss_c = ageLoss(age_pred, label_age)
		loss_c_rev = ageLoss(age_pred_r, tf.ones_like(label_age)/AGE_CATEGORY)

		# ae loss
		loss_ae_d = ageLoss(age_real, label_age)
		loss_ae_g = ageLoss(age_fake_att, label_age) + ageLoss(age_fake, label_age)

		losses = [loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g]
		weights = [1., 0.001, 0.001, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1]
		reweighted_losses = [w*l for w,l in zip(weights, losses)]

	return losses, reweighted_losses, tape

def applyGrad(losses, AIM, optim, tape):
	# loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g = losses

	var_AutoEencoder = AIM.encoder.variables + AIM.decoder.variables
	var_E = AIM.encoder.variables
	var_G = AIM.decoder.variables
	var_dz = AIM.dis_z.variables
	var_dimg = AIM.dis_img.variables
	var_age = AIM.age_classifier.variables

	variables = [var_AutoEencoder, var_dz, var_E, var_dimg, var_G, var_age, var_age, var_dz, var_G]
	grads = tape.gradient(losses, variables)
	# grads = tape.gradient(losses[0], variables[0])

	# optim.apply_gradients(zip(grads, variables[0]))

	for g,v in zip(grads, variables):
		optim.apply_gradients(M.zip_grad(g,v))

def printLosses(losses, i, eta):
	loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g = losses

	print('ITER:%d\tIMG:%.4f\tDZ:%.4f\tGZ:%.4f\tDIMG:%.4f\tGIMG:%.4f\tC1:%.4f\tC2:%.4f\tAE:%.4f\tAE2:%.4f\tETA:%s'%\
		(i, loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g, eta.get_ETA(i)))
