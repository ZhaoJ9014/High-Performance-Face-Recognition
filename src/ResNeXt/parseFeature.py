import scipy.io as sio
import pickle
import numpy as np
import os
import numpy as np
import time


path = "/media/zhaojian/6TB/data/tensorflow_model_result/siamese/"
outputDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/vggBaseTestSet/"
features = sio.loadmat(path + "basetest.mat")
data = features['data']
labels = features['label']
# print "labels[0]: ", labels[0]
for index in range(len(data)):
	personData = np.asarray(data[index])
	label = str(labels[:, index][0][0])
	print "label: ", label.split(" ")
	print "personData.shape: ", personData.shape
	sio.savemat(outputDir + label, {"identityFeature": personData})
