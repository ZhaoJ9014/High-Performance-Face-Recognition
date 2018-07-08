import scipy.io as sio
import pickle
import numpy as np
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import spatial
from sklearn.externals import joblib
import time

reducedDim = 2048
pca = PCA(n_components = reducedDim, whiten = True)

path = "/media/zhaojian/6TB/data/extra_general_model_feature/"

with open(path + "NovelSet_List/NovelSet_1.txt", 'r') as f:
	lines = f.readlines()

vggFeatures = np.loadtxt(path + 'NovelSet_Fea/VGG_NOVELSET_1.txt')
print "vggFeatures.shape: ", vggFeatures.shape

inputFeaturePath = "extracted_feature/NovelSet_1IdentityFeature/"
outputFeaturePath = "extracted_feature/NovelSet_1IdentityFeaturePCA2048/"

features = []
labelList = []
for index in range(len(lines)):
	print index
	line = lines[index]
	ID = line.split("/")[-2]
	print ID
	labelList.append(ID)

	vggFeature = feature = vggFeatures[index].flatten()
	print "vggFeature.shape", vggFeature.shape

	# caffeFeature = sio.loadmat(inputFeaturePath + ID + ".mat")["identityFeature"].flatten()
	# print "caffeFeature.shape", caffeFeature.shape
	#
	# identityFeature = np.concatenate((caffeFeature, vggFeature), axis = 0)
	# print "identityFeature.shape: ", identityFeature.shape

	identityFeature = vggFeature
	features.append(identityFeature)

features = np.asarray(features)
print "features..shape: ", features.shape

# sio.savemat("concatenateFeatures", {"identityFeature": features})
# sio.savemat("vggNovelSet_1_Features", {"identityFeature": features})
features = sio.loadmat("vggNovelSet_1_Features")['identityFeature']

#
# features = pca.fit_transform(features)
#
print "features..shape: ", features.shape
#
#
for index in range(len(features)):
	identityFeature = features[index]
	
	print "identityFeature.shape: ", identityFeature.shape
	label = labelList[index]
	# print index
	# print label
	sio.savemat(outputFeaturePath + label, {"identityFeature": identityFeature})
