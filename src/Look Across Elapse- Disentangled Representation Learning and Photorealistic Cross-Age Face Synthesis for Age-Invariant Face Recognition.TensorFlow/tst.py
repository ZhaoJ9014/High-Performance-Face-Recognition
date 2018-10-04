import tensorflow as tf
from FaceAging import FaceAging
from ops import *
from scipy.misc import imread, imresize, imsave

import model as M 
M.set_gpu('0')

import os 
if not os.path.exists('./res/'):
	os.mkdir('./res/')

sess = tf.Session()
net = FaceAging(sess, is_training=False, size_batch=5)
net.load_checkpoint()

def generate(img, sample_gender,pref):
	for i in range(10):
		age = np.ones([len(img),10],dtype=np.float32)
		age = age * -1.
		age[:,i] = 1.
		gen = net.session.run(net.G, feed_dict={net.input_image:img, net.age:age, net.gender:sample_gender})
		gen = gen + 1.
		gen = gen * 0.5
		imgbuff = (np.float32(img) +1.)*0.5
		for j in range(len(img)):
			imsave('./res/%s_%d_%i.jpg'%(pref[j],j,i),gen[j])
			imsave('./res/%s_%d.jpg'%(pref[j],j),imgbuff[j])

f = open('list.txt')
img_fnames = []
sample_gender = []
preflist = []
cnt = 0
for i in f:
	i = i.strip()
	gend = i.replace('\\','/').split('/')[-1].split('.')[0].split('_')[1]
	sample_gender.append(int(gend))
	img_fnames.append(i)
	pref = i.replace('\\','/').split('/')[-1].split('.')[0].split('_')[2] + '_' + str(cnt)
	cnt += 1
	preflist.append(pref)

	if len(img_fnames)==5:
		imgs = [load_image(image_path=i, image_size=net.size_image, image_value_range=(-1,1)) for i in img_fnames]
		eye2 = np.eye(2,dtype=np.float32)
		sample_gender = eye2[sample_gender]
		sample_gender = sample_gender * 2. - 1.
		generate(imgs, sample_gender, preflist)
		img_fnames = []
		sample_gender = []
		preflist = []
