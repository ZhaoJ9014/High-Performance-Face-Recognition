import scipy.io as sio
import pickle
import numpy as np
import os

def generateNovelSetIdentityFeatures():
	name = "lowshotImg_cropped5_224"
	# path = "extracted_feature/MSchallenge2BaseIdentityMeanFeature/"
	path = "extracted_feature/" + name + "IdentityFeature/"

	if not os.path.isdir(path):
		os.mkdir(path)

	# NumtoID = pickle.load(open("MSchallenge2lowshot_224_NumtoID.p", "rb"))
	labelList = pickle.load(open(name + "LabelList.p", "rb"))
	# NumtoID = pickle.load(open("MSchallenge2Base_NumtoID.p", "rb"))
	# labelList = pickle.load(open("MSchallenge2BaseLabelList.p", "rb"))

	# print len(NumtoID)
	print len(labelList)
	print "loading features....."
	featureList = np.loadtxt('extracted_feature/' + name + '_Feature.txt')
	print "finish loading features....."
	# featureList = sio.loadmat('extracted_feature/MSchallenge2lowshot_224_feature.mat')["train_features_resnext_s"]
	print featureList.shape
	print labelList[:10]
	preLabel = labelList[0]
	features = []
	for index in range(len(labelList)):
		label = labelList[index]
		print "label: ", label
		feature = featureList[index]
		print "feature.shape: ", feature.shape
		if label == preLabel:
			features.append(feature)
		else:
			identityFeature = np.asarray(features)
			# identityFeature = np.mean(features, axis = 0)
			print "identityFeature.shape: ", identityFeature.shape
			sio.savemat(path + preLabel, {"identityFeature": identityFeature})
			preLabel = label
			features = []
			features.append(feature)

	features = np.asarray(features)
	# identityFeature = np.mean(features, axis = 0)
	print "identityFeature.shape: ", identityFeature.shape
	sio.savemat(path + preLabel, {"identityFeature": identityFeature})

# def testCosineSimilarity():
generateNovelSetIdentityFeatures()
