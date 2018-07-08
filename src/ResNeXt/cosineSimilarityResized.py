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
# from matplotlib.mlab import PCA

class TestCosineSimilarity(object):
	def __init__(self):
		self.time = time.strftime("%Y-%m-%d-%H-%M")
		# self.name = "C2test"
		# self.name = "MSchallenge2Base"
		# self.name = "lowshotImg_cropped5_224"
		self.reducedDim = 2048
		# self.reducedDim = 512
		# self.reducedDim = 1024


		# self.tag = ""
		self.tag = ""

		self.pca = PCA(n_components = self.reducedDim, whiten = True)
		self.standard_scaler = StandardScaler()

		# self.identityFeatureDir = "extracted_feature/lowshotImg_cropped5_224MeanFeature/"
		# self.PCAIdentityFeatureDir = "extracted_feature/lowshotImg_cropped5_224MeanFeaturePCA/"
		# self.identityFeatureDir = "extracted_feature/" + self.name + "IdentityFeature/"
		# self.PCAIdentityFeatureDir = "extracted_feature/" + self.name + "IdentityFeaturePCA/"

		# self.totalIdentityFeatureDir = "extracted_feature/Challenge2FeatureResized5/"
		# self.totalIdentityFeatureDir = "extracted_feature/Challenge2Feature_test/"
		self.totalIdentityFeatureDir = "extracted_feature/Challenge2Feature/"


		# self.totalIdentityFeatureDir = "extracted_feature/Challenge2FeatureTest/"


		# self.totalIdentityFeatureDir = "extracted_feature/MSchallenge2BaseIdentityMeanFeature/"
		self.testDataPCAdir = "extracted_feature/C2testResized_featurePCA" + str(self.reducedDim) + self.tag + "/"

		# self.totalPCAidentityFeatureDir = "extracted_feature/Challenge2FeaturePCA/"
		self.PCAIdentityFeatureDir = "extracted_feature/totalIdentityFeaturePCA" + str(self.reducedDim) + self.tag + "/"

		if not os.path.isdir(self.testDataPCAdir):
			os.mkdir(self.testDataPCAdir)
		if not os.path.isdir(self.PCAIdentityFeatureDir):
			os.mkdir(self.PCAIdentityFeatureDir)

		# self.labelList = pickle.load(open(self.name + "LabelList.p", "rb"))
		# print len(self.labelList)



	def generateNovelSetIdentityFeatures(self):
		print "generateNovelSetIdentityFeatures"
		name = "lowshotImg_cropped_224_test"
		path = "extracted_feature/" + name + "IdentityFeatureResized/"
		# path = "extracted_feature/Challenge2FeatureTest/"

		if not os.path.isdir(path):
			os.mkdir(path)

		labelList = pickle.load(open(name + "LabelList.p", "rb"))

		print len(labelList)
		print "loading features....."
		featureList = np.loadtxt('extracted_feature/' + name + '_feature_batch/' + name + '_Feature.txt')
		print "finish loading features....."
		# featureList = sio.loadmat('extracted_feature/MSchallenge2lowshot_224_feature.mat')["train_features_resnext_s"]
		print featureList.shape
		print labelList[:10]
		if len(labelList) != len(featureList):
			raise "len(labelList) != len(featureList)"

		preLabel = labelList[0]
		features = []
		for index in range(len(featureList)):
			print "generateNovelSetIdentityFeatures"
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

	def generateBaseSetIdentityFeatures(self):
		print "generateBaseSetIdentityFeatures"
		name = "baseImage_224"

		path = "extracted_feature/" + name + "IdentityFeatureResized/"
		# path = "extracted_feature/Challenge2FeatureTest/"

		if not os.path.isdir(path):
			os.mkdir(path)

		labelList = pickle.load(open(name + "LabelList.p", "rb"))

		print len(labelList)
		print "loading features....."
		# featureList = np.loadtxt('extracted_feature/' + name + '_Feature.txt')

		maxIter = 65
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

	def reducedIdentityDimTrainData(self, name):
		print "reducedIdentityDimTrainData    " + name
		identityFeatureDir = "extracted_feature/" + name + "IdentityFeatureResized/"

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
		print "fitting new pca"
		features = self.standard_scaler.fit_transform(features)
		# pca.fit_transform(x_std)
		self.pca.fit(features)
		# joblib.dump(self.pca,  name +'PCA' + str(self.reducedDim) + self.tag + '.pkl')
		features = self.pca.transform(features)

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

		name = "C2testResized"
		maxIter = 23
		batches = []
		# labelList = pickle.load(open(name + "LabelList.p", "rb"))
		# print "len(labelList): ", len(labelList)

		for iter in range(maxIter + 1):
			print "reducedIdentityDimTestData"
			print "iter_" + str(iter)
			print "loading features....."
			print 'extracted_feature/C2testResized_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/C2testResized_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt')
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

		batches = self.standard_scaler.transform(batches)
		self.pca.fit(batches)
		# joblib.dump(self.pca,  name +'PCA' + str(self.reducedDim) + '.pkl')
		batches = self.pca.transform(batches)

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
		name = "C2testResized"
		maxIter = 23
		batches = []

		for iter in range(maxIter + 1):
			print "reducedIdentityDimTestData"
			print "iter_" + str(iter)
			print "loading features....."
			print 'extracted_feature/C2testResized_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/C2testResized_feature_batch/' + name + '_feature_batch' + str(iter) + '.txt')
			print "batch.shape: ", batch.shape
			print "finish loading features....."
			batches.extend(batch)
		batches = np.float32(batches)
		print "testCosineSimilarity"
		testFeatures = batches/np.linalg.norm(batches,axis=1,keepdims=True)

		print "testFeatures.shape: ", testFeatures.shape

		galleryFeatures = []
		identities = os.listdir(self.totalIdentityFeatureDir)
		print identities[:10]
		labels = []
		for identity in identities:
			# identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"]
			identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"].flatten()
			galleryFeatures.append(identityFeature)
			labels.append(identity)

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
			# print "pred: ", pred
			preds.append(pred)

		name = 'mxnetPred' + self.tag +  self.time

		sio.savemat(name,{'data':preds,'scr':scr})
		lines = sio.loadmat(name)['data']
		print "len(lines): ", len(lines)

		name = "C2test"
		labelList = pickle.load(open(name + "LabelList.p", "rb"))
		print len(labelList)



		result = []
		for index in range(len(lines)):
		    line = lines[index]
		    label = labelList[index]
		    # print "label: ", label
		    # print "line: ", line
		    label = label.replace(".mat", "")
		    pred = line.split(" ")[0].replace(".mat", "")
		    if pred == label:
		        result.append(1)
		    else:
		        result.append(0)

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
		# self.reducedIdentityDimTrainData("lowshotImg_cropped5_224")
		# self.reducedIdentityDimTrainData("MSchallenge2Base")
		# self.reducedIdentityDimTestData()
		# self.testCosineSimilarityPCA()
		# self.testCosineSimilarity()


if __name__ == '__main__':
	tcs = TestCosineSimilarity()
	tcs.run()
