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
import sys
sys.path.append('/home/zhaojian/liblinear/python')
from liblinearutil import *
from scipy import sparse

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__

# from matplotlib.mlab import PCA

class TestSimilarity(object):
	def __init__(self):
		self.time = time.strftime("%Y-%m-%d-%H-%M")
		self.reducedDim = 2048
		self.tag = "NovelSet_1"

		# self.pca = PCA(n_components = self.reducedDim, whiten = True)
		# self.standard_scaler = StandardScaler()

		# self.identityFeatureDir = "extracted_feature/" + self.name + "IdentityFeature/"
		# self.PCAIdentityFeatureDir = "extracted_feature/" + self.name + "IdentityFeaturePCA/"

		self.totalIdentityFeatureDir = "extracted_feature/Challenge2Feature_Novel_1_Mean/"
		# self.totalTestFeatureDIr = "MS_dev_set/"
		# self.totalIdentityFeatureDir = "extracted_feature/totalIdentityFeaturePCA_matlab/"

		# self.testDataPCAdir = "extracted_feature/C2test_featurePCA" + str(self.reducedDim) + self.tag + "/"

		# self.totalPCAidentityFeatureDir = "extracted_feature/Challenge2FeaturePCA/"
		# self.PCAIdentityFeatureDir = "extracted_feature/totalIdentityFeaturePCA" + str(self.reducedDim) + self.tag + "/"

		# if not os.path.isdir(self.testDataPCAdir):
		# 	os.mkdir(self.testDataPCAdir)
		# if not os.path.isdir(self.PCAIdentityFeatureDir):
		# 	os.mkdir(self.PCAIdentityFeatureDir)



	def generateNovelSetIdentityFeatures(self):
		print "generateNovelSetIdentityFeatures"
		# name = "lowshotImg_cropped5_224"
		# name = "NovelSet_1_generated"
		name = "JK_BaseSet_first_part"

		path = "extracted_feature/" + name + "_IdentityFeatureMean/"

		if not os.path.isdir(path):
			os.mkdir(path)

		print "loading features....."
		# featureList = np.loadtxt('extracted_feature/' + name + "_feature_batch/" + name + '_Feature.txt')
		featureList = np.loadtxt("/home/zhaojian/DEEP/JK_GoogleNet_BN/features/" + name + ".txt")

		print "finish loading features....."
		print featureList.shape

		labelList = pickle.load(open('extracted_feature/' + name + "LabelList.p", "rb"))
		print len(labelList)
		print labelList[:10]

		if len(labelList) != len(featureList):
			labelLength = len(labelList)
			featureList = featureList[:len(labelList)]
			# raise "len(labelList) != len(featureList)"
			print "len(labelList) != len(featureList)"
			print "cropping the featureList------------------------------------"

		preLabel = labelList[0]
		features = []
		for index in range(len(featureList)):
			print "generateNovelSetIdentityFeatures"
			label = labelList[index]
			# print "label: ", label
			feature = featureList[index]
			# print "feature.shape: ", feature.shape
			if label == preLabel:
				features.append(feature)
			else:
				features = np.asarray(features)
				print "preLabel: ", preLabel
				print "features.shape: ", features.shape
				identityFeature = np.mean(features, axis = 0)
				# identityFeature = features

				print "identityFeature.shape: ", identityFeature.shape
				sio.savemat(path + preLabel, {"identityFeature": identityFeature})
				preLabel = label
				features = []
				features.append(feature)

		features = np.asarray(features)
		print "preLabel: ", preLabel
		print "features.shape: ", features.shape
		identityFeature = np.mean(features, axis = 0)

		print "identityFeature.shape: ", identityFeature.shape
		sio.savemat(path + preLabel, {"identityFeature": identityFeature})

	def generateBaseSetIdentityFeatures(self):
		print "generateBaseSetIdentityFeatures"
		name = "BaseSet"

		path = "extracted_feature/" + name + "IdentityFeature/"

		if not os.path.isdir(path):
			os.mkdir(path)

		labelList = pickle.load(open("extracted_feature/" + name + "LabelList.p", "rb"))

		print len(labelList)
		print "loading features....."
		# featureList = np.loadtxt('extracted_feature/' + name + '_Feature.txt')

		maxIter = 231
		batches = []
		for iter in range(maxIter + 1):
			print "generateBaseSetIdentityFeatures"
			print "iter_" + str(iter)
			print "loading features....."
			print 'extracted_feature/' + name + '_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/' + name + '_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt')
			print "batch.shape: ", batch.shape
			print "finish loading features....."
			batches.extend(batch)

		featureList = np.asarray(batches)
		print "finish loading features....."
		# featureList = sio.loadmat('extracted_feature/MSchallenge2lowshot_224_feature.mat')["train_features_resnext_s"]
		print featureList.shape
		print labelList[:10]
		if len(labelList) != len(featureList):
			raise "len(labelList) != len(featureList)"

		preLabel = labelList[0]
		features = []
		for index in range(len(featureList)):
			print "generateBaseSetIdentityFeatures"
			label = labelList[index]
			print "label: ", label
			feature = featureList[index]
			print "feature.shape: ", feature.shape
			if label == preLabel:
				features.append(feature)
			else:
				features = np.asarray(features)

				identityFeature = np.mean(features, axis = 0)

				print "identityFeature.shape: ", identityFeature.shape
				sio.savemat(path + preLabel, {"identityFeature": identityFeature})
				preLabel = label
				features = []
				features.append(feature)

		features = np.asarray(features)

		identityFeature = np.mean(features, axis = 0)

		print "identityFeature.shape: ", identityFeature.shape
		sio.savemat(path + preLabel, {"identityFeature": identityFeature})


	def generateBaseSetIdentityFeaturesMemoryFriendly(self):
		# NumtoID = pickle.load(open("MSchallenge2lowshot_224_NumtoID.p", "rb"))
		# labelList = pickle.load(open("MSchallenge2lowshot_224LabelList.p", "rb"))
		# NumtoID = pickle.load(open(name + "_NumtoID.p", "rb"))
		# print len(NumtoID)
		chunk = 5000
		maxIter = 231
		features = []
		preFeatures = []
		preLabel = None
		name = "BaseSet"

		path = "extracted_feature/" + name + "IdentityFeature/"
		if not os.path.isdir(path):
			os.mkdir(path)

		totalLabelList = pickle.load(open("extracted_feature/" + name + "LabelList.p", "rb"))

		for iter in range(maxIter + 1):
			print "loading features....."
			print 'extracted_feature/' + name + '_feature/' + name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/' + name + '_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt')
			print "finish loading features....."
			print "iter_" + str(iter), " ", batch.shape

			if iter == maxIter:
				labelList = totalLabelList[iter * chunk : ]
			else:
				labelList = totalLabelList[iter * chunk : (iter + 1) * chunk]

			print "len(batch): ", len(batch)
			print "len(labelList): ", len(labelList)
			if len(labelList) != len(batch):
				raise "len(labelList) != len(batch)"

			if len(preFeatures) != 0:
				features = preFeatures
			else:
				preLabel = labelList[0]
				features = []

			for index in range(len(labelList)):
				label = labelList[index]
				# print "label: ", label
				feature = batch[index]
				# print "feature.shape: ", feature.shape
				if label == preLabel:
					features.append(feature)
				else:
					identityFeature = np.asarray(features)
					identityFeature = np.mean(features, axis = 0)
					print "identityFeature.shape: ", identityFeature.shape
					sio.savemat(path + preLabel, {"identityFeature": identityFeature})
					print "save: ", path + preLabel
					preLabel = label
					features = []
					features.append(feature)
					preFeatures = []

			if len(features) != 0 and iter != maxIter:
				preFeatures = features
			else:
				features = np.asarray(features)
				identityFeature = np.mean(features, axis = 0)
				print "identityFeature.shape: ", identityFeature.shape
				sio.savemat(path + preLabel, {"identityFeature": identityFeature})
				print "save: ", path + preLabel

	def reducedIdentityDimTrainData(self, name):
		print "reducedIdentityDimTrainData    " + name
		identityFeatureDir = "extracted_feature/" + name + "IdentityFeature/"

		# labelList = pickle.load(open(name + "LabelList.p", "rb"))
		# print "len(labelList): ", len(labelList)
		labelList = []
		identities = os.listdir(identityFeatureDir)
		print "len(identities): ", len(identities)
		features = []
		for identity in identities:
			# print "reducedIdentityDimTrainData    " + name
			# print "identity: ", identity
			feature = sio.loadmat(identityFeatureDir + identity)["identityFeature"].flatten()
			# .flatten()
			print "feature.shape: ", feature.shape
			# (num, dim) = feature.shape
			# if num  < 1 or dim != 2048:
			# 	raise "num  < 1 or dim != 2048"
			features.append(feature)
			labelList.append(identity)

		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		print "reducedIdentityDimTrainData    " + name
		features = np.asarray(features)
		print "len(labelList): ", len(labelList)
		print "features.shape: ", features.shape

		if len(labelList) != len(features):
			raise "len(labelList) != len(features)"

		# features = self.pca.fit_transform(features)

		# if os.path.isfile(name +'PCA' + str(self.reducedDim) + self.tag + '.pkl'):
		# 	print "loading exisitng pca"
		# 	self.pca = joblib.load(name +'PCA' + str(self.reducedDim) + self.tag + '.pkl')
		# else:

		# features = PCA(features)
		# print "fitting new pca"
		# features = self.standard_scaler.fit_transform(features)
		# # pca.fit_transform(x_std)
		# self.pca.fit(features)
		# # joblib.dump(self.pca,  name +'PCA' + str(self.reducedDim) + self.tag + '.pkl')
		# features = self.pca.transform(features)

		print "after PCA features.shape: ", features.shape
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
		# np.savetxt('extracted_feature/' + self.name + 'PCAfeature.txt', features)
		# raise "finish saving"

		for index in range(len(features)):
			# print "reducedIdentityDimTrainData    " + name
			label = labelList[index]
			feature = features[index]

			sio.savemat(self.PCAIdentityFeatureDir + label, {"identityFeature": feature})
			print "save: ", self.PCAIdentityFeatureDir + label

	def reducedIdentityDimTestData(self):

		name = "C2test"
		maxIter = 23
		batches = []
		# labelList = pickle.load(open(name + "LabelList.p", "rb"))
		# print "len(labelList): ", len(labelList)

		for iter in range(maxIter + 1):
			print "reducedIdentityDimTestData"
			print "iter_" + str(iter)
			print "loading features....."
			print 'extracted_feature/C2test_feature/' + name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/C2test_feature/' + name + '_feature_batch' + str(iter) + '.txt')
			print "batch.shape: ", batch.shape
			print "finish loading features....."
			batches.extend(batch)

		batches = np.asarray(batches)
		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		print "batches.shape: ", batches.shape

		# if len(labelList) != len(batches):
		# 	raise "len(labelList) != len(features)"

		# if os.path.isfile(name +'PCA' + str(self.reducedDim) + '.pkl'):
		# 	self.pca = joblib.load(name +'PCA' + str(self.reducedDim) + '.pkl')
		# else:

		# batches = PCA(batches)
		#
		# batches = self.standard_scaler.fit_transform(batches)
		# # batches = self.standard_scaler.transform(batches)
		# self.pca.fit(batches)
		# # joblib.dump(self.pca,  name +'PCA' + str(self.reducedDim) + '.pkl')
		# batches = self.pca.transform(batches)

		print "after PCA batches.shape: ", batches.shape
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"

		counter = 0
		for index in range(len(batches)):
			# label = labelList[index]
			feature = batches[index]
			counter += 1
			sio.savemat(self.testDataPCAdir + str(counter), {"identityFeature": feature})
			# print label
			if counter % 100 == 0:
				print "reducedIdentityDimTestData counter: ", counter


	def writeToFile(self, content, name):
		with open(name, 'a') as f:
			f.write(content)

	def testCosineSimilarityPCA(self):
		print "testCosineSimilarityPCA"
		testFeatures = []
		testIdentities = os.listdir(self.testDataPCAdir)
		totalTestIdentityNum = len(testIdentities)
		# labelList = []
		for index  in range(totalTestIdentityNum):
			testIdentity = str(index + 1)
			# print "testCosineSimilarityPCA"
			# testIdentity = testIdentities[index]
			# labelList.append(testIdentity)
			# print "totalTestIdentityNum: ", totalTestIdentityNum
			# print "testIdentity index: ", index
			testIdentityFeature = sio.loadmat(self.testDataPCAdir + testIdentity)["identityFeature"].flatten()
			# print "totalTestIdentityNum: ", totalTestIdentityNum
			# print "testIdentityFeature.shape: ", testIdentityFeature.shape
			testFeatures.append(testIdentityFeature)
		testFeatures = np.float32(testFeatures)
		testFeatures = testFeatures/np.linalg.norm(testFeatures,axis=1,keepdims=True)

		print "testFeatures.shape: ", testFeatures.shape

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
		# print "testCosineSimilarity"
		# testFeatures = batches/np.linalg.norm(batches,axis=1,keepdims=True)
		#
		# print "testFeatures.shape: ", testFeatures.shape

		galleryFeatures = []
		# identities = os.listdir(self.totalIdentityFeatureDir)
		identities = os.listdir(self.PCAIdentityFeatureDir)
		print identities[:10]
		for identity in identities:
			# identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"].flatten()
			identityFeature = sio.loadmat(self.PCAIdentityFeatureDir + identity)["identityFeature"].flatten()
			print "identityFeature.shape: ", identityFeature.shape
			galleryFeatures.append(identityFeature)

		galleryFeatures = np.float32(galleryFeatures)
		galleryFeatures = galleryFeatures/np.linalg.norm(galleryFeatures,axis=1,keepdims=True)

		print "galleryFeatures.shape: ", galleryFeatures.shape
		galleryFeatures = galleryFeatures.transpose()
		print "galleryFeatures.shape: ", galleryFeatures.shape

		print('Computing the result...')
		a = time.time()
		result = np.dot(testFeatures,galleryFeatures)
		b = time.time()
		print('time elapsed:',b-a)
		print(result.shape)
		scr = np.amax(result,axis=1)
		result = np.argmax(result,axis=1)

		preds = []
		for index in result:
			pred = identities[index]
			print "pred: ", pred
			preds.append(pred)

		name = 'mxnetPredPCA' + str(self.reducedDim) + self.tag +  self.time
		sio.savemat(name,{'data':preds,'scr':scr})
		# pickle.dump( labelList, open( "C2testLabelListPCA" + str(self.reducedDim) + self.tag +  self.time + ".p", "wb" ) )

	def testCosineSimilarity(self):
		print "testCosineSimilarity"
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

		# name = "DevNovelSet"
		# name = "learnnetBaseTestSet"
		# name = "vggBaseTestSet"
		name = "novelTestList"

		testSetName = name
		# ---------------------------------------------
		# testIdentityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + name + "/"
		# testIdentities =os.listdir(testIdentityFeatureDir)
		# testFeatures = []
		# for identity in testIdentities:
		# 	print identity
		# 	identityFeature = sio.loadmat(testIdentityFeatureDir + identity)["identityFeature"].flatten()
		# 	print "testCosineSimilarity"
		# 	print "identityFeature.shape: ", identityFeature.shape
		# 	testFeatures.append(identityFeature)
		# batches = testFeatures
		# testLabelList = testIdentities
		# ---------------------------------------------
		testLabelList = pickle.load(open('extracted_feature/' + name + "LabelList.p", "rb"))
		# f = open('extracted_feature/' + name + "LabelList.txt", 'r')
		# testLabelList = f.readlines()
		print "testLabelList[0]: ", testLabelList[0].split(" ")
		print len(testLabelList)
		batches = np.loadtxt('extracted_feature/' + name  + "_feature_batch/" + name + '_Feature.txt')
		# batches = sio.loadmat('extracted_feature/' + name  + "_feature_batch512/" + name + '_Feature.mat')['features']
		# ---------------------------------------------
		# name2 = "DevBaseSet"
		# f2 = open('extracted_feature/' + name2 + "LabelList.txt", 'r')
		# testLabelList2 = f2.readlines()
		# print "testLabelList[0]: ", testLabelList2[0].split(" ")
		# print len(testLabelList2)
		# batches2 = np.loadtxt('extracted_feature/' + name2  + "_feature_batch/" + name2 + '_Feature.txt')
		# # batches = sio.loadmat('extracted_feature/' + name  + "_feature_batch512/" + name + '_Feature.mat')['features']
		# testLabelList = testLabelList + testLabelList2
		# batches = np.concatenate((batches, batches2), axis = 0)

		# ---------------------------------------------
		print "testCosineSimilarity"

		batches = np.float32(batches)
		print "batches.shape: ", batches.shape

		testFeatures = batches/np.linalg.norm(batches,axis=1,keepdims=True)
		print "testFeatures.shape: ", testFeatures.shape

		galleryIdentities = []
		#-----------------------------------------------------------------------
		# identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/BaseSetIdentityFeatureMean/"
		# name  = "learnnetBaseSet"
		# name  = "vggBaseSet"
		name  = "BaseSetIdentityFeatureMean"
		identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + name + "/"

		identities = os.listdir(identityFeatureDir)
		print identities[:10]
		galleryIdentities.extend(identities)

		galleryFeatures = []
		for identity in identities:
			# identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"]
			print identity

			identityFeature = sio.loadmat(identityFeatureDir + identity)["identityFeature"].flatten()
			# identityFeature = np.mean(identityFeature, axis = 0).flatten()

			# identityFeature = sio.loadmat(identityFeatureDir + identity)['tmp_1']["identityFeature"][0][0][0].flatten()

			# print "BaseSetIdentityFeatureMean"
			# print "identityFeature.shape: ", identityFeature.shape
			galleryFeatures.append(identityFeature)
		#-----------------------------------------------------------------------
		# name  = "vggNovelSet5"
		# name  = "learnnetNovelSet_5"
		name = "NovelSet_1IdentityFeatureMean"
		identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + name + "/"

		#
		# identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + self.tag + "IdentityFeaturePCA2048/"
		# identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + self.tag + "IdentityFeatureMean/"
		# identityFeatureDir = "/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/" + self.tag + "IdentityFeature512/"

		identities =os.listdir(identityFeatureDir)
		# identities =os.listdir(self.totalIdentityFeatureDir)
		print identities[:10]
		galleryIdentities.extend(identities)

		for identity in identities:
			# identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"].flatten()
			identityFeature = sio.loadmat(identityFeatureDir + identity)["identityFeature"].flatten()
			#
			# identityFeature = sio.loadmat(identityFeatureDir + identity)['tmp_1']["identityFeature"][0][0][0].flatten()
			# print self.tag + "IdentityFeaturePCA2048"
			# print "identityFeature.shape: ", identityFeature.shape

			galleryFeatures.append(identityFeature)
		#-----------------------------------------------------------------------
		galleryFeatures = np.float32(galleryFeatures)
		galleryFeatures = galleryFeatures/np.linalg.norm(galleryFeatures,axis=1,keepdims=True)

		print "galleryFeatures.shape: ", galleryFeatures.shape
		galleryFeatures = galleryFeatures.transpose()
		print "galleryFeatures.shape: ", galleryFeatures.shape

		print('Computing the result...')
		a = time.time()
		result = np.dot(testFeatures,galleryFeatures)
		b = time.time()
		print('time elapsed:',b-a)
		print(result.shape)

		scr = np.amax(result,axis=1)
		result = np.argmax(result,axis=1)

		preds = []
		for index in result:
			pred = galleryIdentities[index]
			# print "pred: ", pred
			preds.append(pred)


		lines = preds
		# lines = sio.loadmat(name)['data']
		print "len(lines): ", len(lines)

		result = []
		for index in range(len(lines)):
			line = lines[index]
			label = testLabelList[index].replace("\n", "")
			# print "label: ", label
			# print "line: ", line
			label = label.replace(".mat", "")
			pred = line.split(" ")[0].replace(".mat", "")
			print "vvvvvvvvvvvvvvvvvvvvvvvvv"
			print "label: ", label
			print "pred: ", pred
			print "^^^^^^^^^^^^^^^^^^^^^^^^^"
			if pred == label:
				result.append(1)
			else:
				result.append(0)

		name = 'pred_' + testSetName + "_" + self.tag + "_" + self.time

		sio.savemat('extracted_feature/' + name,{'preds':preds, 'result': result, 'scr':scr})

		print "Accuracy: ", sum(result)/float(len(result))

		C = sum(result)
		N = len(lines)
		M_099 = C / 0.99
		M_095 = C / 0.95

		Coverage_099 = M_099 / N
		Coverage_095 = M_095 / N

		print "Coverage_099: ", Coverage_099
		print "Coverage_095: ", Coverage_095
		print self.totalIdentityFeatureDir

	def testSVMSimilarity(self):
		print "testSVMSimilarity"
		name = "DevNovelSet"
		testSetName = name
		# ---------------------------------------------
		# testLabelList = pickle.load(open('extracted_feature/' + name + "LabelList.p", "rb"))
		f = open('extracted_feature/' + name + "LabelList.txt", 'r')
		testLabelList = f.readlines()
		print "testLabelList[0]: ", testLabelList[0].split(" ")
		print len(testLabelList)
		batches = np.loadtxt('extracted_feature/' + name  + "_feature_batch/" + name + '_Feature.txt')
		# batches = sio.loadmat('extracted_feature/' + name  + "_feature_batch512/" + name + '_Feature.mat')['features']
		# ---------------------------------------------
		# name2 = "DevBaseSet"
		# f2 = open('extracted_feature/' + name2 + "LabelList.txt", 'r')
		# testLabelList2 = f2.readlines()
		# print "testLabelList[0]: ", testLabelList2[0].split(" ")
		# print len(testLabelList2)
		# batches2 = np.loadtxt('extracted_feature/' + name2  + "_feature_batch/" + name2 + '_Feature.txt')
		# # batches = sio.loadmat('extracted_feature/' + name  + "_feature_batch512/" + name + '_Feature.mat')['features']
		# testLabelList = testLabelList + testLabelList2
		# 	batches = np.concatenate((batches, batches2), axis = 0)

		# ---------------------------------------------
		print "testSVMSimilarity"
		# batches = np.float32(batches)
		print "batches.shape: ", batches.shape
		testFeatures = batches
		# testLabelList = testLabelList[:1]
		testFeatures = batches/np.linalg.norm(batches,axis=1,keepdims=True)
		print "testFeatures.shape: ", testFeatures.shape

		#-----------------------------------------------------------------------
		svmDir = "/media/zhaojian/6TB/project/Face_Cluster_SVM_Classification/norm_svmModel/"

		svms = os.listdir(svmDir)

		preds = []
		a = time.time()
		for index in range(len(testFeatures)):
			testFeature = np.asarray(testFeatures[index])
			testFeature = testFeature[np.newaxis, :]
			testFeature = sparse.csr_matrix(testFeature)
			featurePreds = []
			blockPrint()
			for svm in svms:
				# print "svm: ", svm
				m = load_model(svmDir + svm)
				# print "testFeature.shape: ", testFeature.shape
				p_labels, p_acc, p_vals = predict([], testFeature, m)
				featurePreds.append(p_labels[0])
			preds.append(featurePreds)
			enablePrint()
			if index % 5 == 0:
				b = time.time()
				print('time elapsed:',b-a)
				a = b
				print "index: ", index
				self.calculateAccuracy(preds, testLabelList, svms, testSetName)

	def calculateAccuracy(self, preds, testLabelList, svms, testSetName):
		result = np.asarray(preds)
		print("result.shape: ", result.shape)
		scr = np.amax(result,axis=1)
		result = np.argmax(result,axis=1)
		print('Computing the result...')

		preds = []
		for index in result:
			pred = svms[index]
			# print "pred: ", pred
			preds.append(pred)

		lines = preds
		# lines = sio.loadmat(name)['data']
		print "len(lines): ", len(lines)

		result = []
		for index in range(len(lines)):
			line = lines[index]
			label = testLabelList[index].replace("\n", "")
			# print "label: ", label
			# print "line: ", line
			label = label.replace(".mat", "")
			pred = line.split(" ")[0].replace(".mat.model", "")
			print "vvvvvvvvvvvvvvvvvvvvvvvvv"
			print "label: ", label
			print "pred: ", pred
			print "^^^^^^^^^^^^^^^^^^^^^^^^^"
			if pred == label:
				result.append(1)
			else:
				result.append(0)

		name = 'pred_' + testSetName + "_" + self.tag + "_" + self.time

		# sio.savemat('extracted_feature/' + name,{'preds':preds, 'result': result, 'scr':scr})

		print "Accuracy: ", sum(result)/float(len(result))

		C = sum(result)
		N = len(lines)
		M_099 = C / 0.99
		M_095 = C / 0.95

		Coverage_099 = M_099 / N
		Coverage_095 = M_095 / N

		print "Coverage_099: ", Coverage_099
		print "Coverage_095: ", Coverage_095
		print self.totalIdentityFeatureDir

	def run(self):
		self.generateNovelSetIdentityFeatures()
		# self.generateBaseSetIdentityFeatures()
		# self.generateBaseSetIdentityFeaturesMemoryFriendly()

		# self.reducedIdentityDimTrainData("lowshotImg_cropped5_224")
		# self.reducedIdentityDimTrainData("MSchallenge2Base")
		# self.reducedIdentityDimTestData()
		# self.testCosineSimilarityPCA()

		# self.testCosineSimilarity()
		# self.testSVMSimilarity()


if __name__ == '__main__':
	tcs = TestSimilarity()
	tcs.run()
