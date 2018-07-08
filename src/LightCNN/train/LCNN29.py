import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import model as M
import numpy as np
from hd5_reader import hd5_reader
import shutil

BSIZE = 400
CLASS = 10000
EPOC = 50

def LCNN29():
	with tf.name_scope('img_holder'):
		img_holder = tf.placeholder(tf.float32, [None, 128, 128, 3])
	with tf.name_scope('lab_holder'):
		lab_holder = tf.placeholder(tf.int64, [None, CLASS])

	mod = M.Model(img_holder, [None, 128, 128, 3])

	mod.conv_layer(5, 96, activation=1)
	mod.maxpooling_layer(2, 2) #pool1

	# 
	a = mod.get_current_layer()
	mod.conv_layer(3, 96, activation=1)
	mod.conv_layer(3, 96, activation=1)
	# print (mod.get_shape())

	mod.sum(a)
	mod.conv_layer(1, 96, activation=1)
	mod.conv_layer(3, 192, activation=1)
	mod.maxpooling_layer(2, 2) #pool2

	# 
	b = mod.get_current_layer()
	mod.conv_layer(3, 96*2, activation=1)
	mod.conv_layer(3, 96*2, activation=1)

	mod.sum(b)
	mod.conv_layer(1, 96, activation=1)
	mod.conv_layer(3, 384, activation=1)
	mod.maxpooling_layer(2, 2) #pool3

	# 
	c = mod.get_current_layer()
	mod.conv_layer(3, 192*2, activation=1)
	mod.conv_layer(3, 192*2, activation=1)

	mod.sum(c)
	mod.conv_layer(1, 384, activation=1)
	mod.conv_layer(3, 256, activation=1)

	# 
	d = mod.get_current_layer()
	mod.conv_layer(3, 128*2, activation=1)
	mod.conv_layer(3, 128*2, activation=1)

	mod.sum(d)
	mod.conv_layer(1, 256, activation=1)
	mod.conv_layer(3, 256, activation=1)
	mod.maxpooling_layer(2, 2) #pool4

	mod.flatten()
	mod.fcnn_layer(512)
	feature_layer = mod.get_current_layer()[0]

	mod.dropout(0.2)
	mod.fcnn_layer(CLASS)
	class_layer = mod.get_current_layer()[0]
	acc = mod.accuracy(tf.argmax(lab_holder,1))

	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lab_holder, logits=class_layer))

	with tf.name_scope('accuracy'):
		accuracy = acc

	return loss, accuracy, img_holder, lab_holder

with tf.variable_scope('LCNN29'):
	loss, acc, img_holder, lab_holder = LCNN29()

with tf.name_scope('Optimizer'):
	train = tf.train.AdamOptimizer(0.00001).minimize(loss)
	# print (train)

# 
model_path = './model/'
log_path = './log/'
list_train = 'hd5_list_train.txt'
list_val = 'hd5_list_val.txt'
f_log = open('./log/log.txt', 'a+')
with tf.Session() as sess:

	writer = tf.summary.FileWriter(log_path, sess.graph)
	sess, epoc, iters = M.loadSess(model_path, sess)
	saver = tf.train.Saver()

	reader = hd5_reader(list_train, list_val, BSIZE, BSIZE)
	ITERS = reader.train_epoc
	count = 0
	for j in range(EPOC):
		for i in range(ITERS):
			x_train, y_train_ = reader.train_nextbatch()
			global BSIZE
			BSIZE = reader.train_bsize
			# print ('BSIZE:', BSIZE)
			y_train = np.zeros([BSIZE,CLASS],dtype=np.int64)
			for index in range(BSIZE):
				y_train[index][y_train_[index]] = 1

			_, ls, ac = sess.run([train, loss, acc], feed_dict={img_holder:x_train, lab_holder:y_train})
			# str1 =' Epoc: ' + str(j) + '\t|Iter: ' + str(i) + '\t|Train_Loss: ' + str(ls) + '\t|Train_Acc: ' + str(ac) 
			str1 =' Epoc: ' + str(j+epoc) + '\t|Iter: ' + str(i+iters) + '\t|Train_Loss: ' + str(ls) + '\t|Train_Acc: ' + str(ac) 
			print(str1)
			f_log.write(str1 + '\n')

			# # 
			if count%100 == 0:
				x_val, y_val_ = reader.val_nextbatch()
				global BSIZE
				BSIZE = reader.val_bsize
				y_val = np.zeros([BSIZE,CLASS],dtype=np.int64)
				for index in range(BSIZE):
					y_val[index][y_val_[index]] = 1

				ls_val = 0
				acc_val = 0
				for n in range(reader.val_data_ITERS):
					ls, ac = sess.run([loss, acc], feed_dict={img_holder:x_val, lab_holder:y_val})
					ls_val += ls
					acc_val += ac
				ls_val = ls_val/reader.val_data_ITERS
				acc_val = acc_val/reader.val_data_ITERS
				# print('Epoc:',j,' |Iter:',i,' |Val_Loss1:',ls_val,' |Val_Acc:',acc_val)
				# str1 =' Epoc: ' + str(j) + '\t|Iter: ' + str(i) + '\t|Val_Loss: ' + str(ls_val) + '\t|Val_Acc: ' + str(acc_val) 
				str1 =' Epoc: ' + str(j+epoc) + '\t|Iter: ' + str(i+iters) + '\t|Val_Loss: ' + str(ls_val) + '\t|Val_Acc: ' + str(acc_val) 
				print(str1)
				f_log.write(str1 + '\n')

			if count%1 == 1000 and count > 0:
				save_path = model_path+'Epoc_'+ str(j+epoc) + '_' + 'Iter_' + str(i+iters) + '.cpkt'
				saver.save(sess, save_path)

				save_path2 = save_path + '.meta'
				save_path3 = save_path + '.index'
				save_path4 = save_path + '.data-00000-of-00001'
				save_path5 = model_path + 'checkpoint'

				shutil.copy(save_path2, save_path2.replace('./model/', './backup/'))
				shutil.copy(save_path3, save_path3.replace('./model/', './backup/'))
				shutil.copy(save_path4, save_path4.replace('./model/', './backup/'))
				shutil.copy(save_path5, save_path5.replace('./model/', './backup/'))

			count += 1
	writer.close()
	# writer = tf.summary.FileWrite(log_path, sess.graph)
	# merge = tf.summary.merge_all()
