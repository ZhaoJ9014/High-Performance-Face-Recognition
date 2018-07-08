import os
import scipy.io as sio
import numpy as np
import pickle

# name = "lowshotImg_cropped5_224_padding_"
name = "Challenge2Feature5"
print "collect identity features    " + name
# identityFeatureDir = "extracted_feature/" + name + "IdentityFeature/"
identityFeatureDir = "extracted_feature/" + name
files = os.listdir(identityFeatureDir)

features = []

collectMatList = []
for file in files:
	if ".DS_Store" not in file:
		print file
		feature = sio.loadmat(identityFeatureDir + "/" + file)["identityFeature"].flatten()
		features.append(feature)
		collectMatList.append(file)

features = np.asarray(features)
collectMatList = np.asarray(collectMatList)

if len(features) != len(collectMatList):
	raise "len(features) != len(collectMatList)"

print "features.shape: ", features.shape

pickle.dump( collectMatList, open( "extracted_feature/" + name + "IdentityFeature.p", "wb" ) )
sio.savemat("extracted_feature/" + name + "IdentityFeature", {"identityFeature": features})






#
#
# print "testCosineSimilarity"
# name = "C2test"
# maxIter = 23
# batches = []
#
# for iter in range(maxIter + 1):
# 	print "reducedIdentityDimTestData"
# 	print "iter_" + str(iter)
# 	print "loading features....."
# 	print 'extracted_feature/C2test_feature/' + name + '_feature_batch' + str(iter) + '.txt'
# 	batch = np.loadtxt('extracted_feature/C2test_feature/' + name + '_feature_batch' + str(iter) + '.txt')
# 	print "batch.shape: ", batch.shape
# 	print "finish loading features....."
# 	batches.extend(batch)
# batches = np.float32(batches)
# sio.savemat("extracted_feature/" + name + "IdentityFeature", {"identityFeature": batches})
