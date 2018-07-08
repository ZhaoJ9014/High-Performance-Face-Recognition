import scipy.io as sio
import pickle
import numpy as np
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

class TestCosineSimilarity(object):
	def __init__(self):
		# self.name = "C2test"
		self.name = "MSchallenge2Base"
		# self.name = "lowshotImg_cropped5_224"
		reducedDim = 512
		self.pca = PCA(n_components = reducedDim, whiten = True)

		# self.identityFeatureDir = "extracted_feature/lowshotImg_cropped5_224MeanFeature/"
		# self.PCAIdentityFeatureDir = "extracted_feature/lowshotImg_cropped5_224MeanFeaturePCA/"
		self.identityFeatureDir = "extracted_feature/" + self.name + "IdentityFeature/"
		self.PCAIdentityFeatureDir = "extracted_feature/" + self.name + "IdentityFeaturePCA/"
		# self.totalIdentityFeatureDir = "extracted_feature/Challenge2MeanFeature/"
		# self.totalIdentityFeatureDir = "extracted_feature/MSchallenge2BaseIdentityMeanFeature/"
		self.testDataPCAdir = "extracted_feature/C2test_featurePCA/"
		self.totalPCAidentityFeatureDir = "extracted_feature/Challenge2FeaturePCA/"

		self.labelList = pickle.load(open(self.name + "LabelList.p", "rb"))
		print len(self.labelList)

		self.path = "extracted_feature/" + self.name + "IdentityFeature/"
		if not os.path.isdir(self.path):
			os.mkdir(self.path)

	def generateIdentityFeatures(self):
		# NumtoID = pickle.load(open("MSchallenge2lowshot_224_NumtoID.p", "rb"))
		# labelList = pickle.load(open("MSchallenge2lowshot_224LabelList.p", "rb"))
		# NumtoID = pickle.load(open(name + "_NumtoID.p", "rb"))
		# print len(NumtoID)
		chunk = 5000
		maxIter = 231
		features = []
		preFeatures = []
		preLabel = None

		for iter in range(maxIter + 1):
			print "loading features....."
			print 'extracted_feature/' + self.name + '_feature/' + self.name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/' + self.name + '_feature/' + self.name + '_feature_batch' + str(iter) + '.txt')
			print "finish loading features....."
			print "iter_" + str(iter), " ", batch.shape

			if iter == maxIter:
				labelList = self.labelList[iter * chunk : ]
			else:
				labelList = self.labelList[iter * chunk : (iter + 1) * chunk]

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
					# identityFeature = np.mean(features, axis = 0)
					print "identityFeature.shape: ", identityFeature.shape
					sio.savemat(self.path + preLabel, {"identityFeature": identityFeature})
					print "save: ", self.path + preLabel
					preLabel = label
					features = []
					features.append(feature)
					preFeatures = []

			if len(features) != 0 and iter != maxIter:
				preFeatures = features
			else:
				features = np.asarray(features)
				# identityFeature = np.mean(features, axis = 0)
				print "identityFeature.shape: ", identityFeature.shape
				sio.savemat(self.path + preLabel, {"identityFeature": identityFeature})
				print "save: ", self.path + preLabel


	def reducedIdentityDimTrainData(self):
		# self.name = "lowshotImg_cropped5_224"
		self.name = "MSchallenge2Base"

		self.identityFeatureDir = "extracted_feature/" + self.name + "IdentityFeature/"
		self.PCAIdentityFeatureDir = "extracted_feature/" + self.name + "IdentityFeaturePCA/"
		self.labelList = pickle.load(open(self.name + "LabelList.p", "rb"))


		print "len(self.labelList): ", len(self.labelList)

		if not os.path.isdir(self.PCAIdentityFeatureDir):
			os.mkdir(self.PCAIdentityFeatureDir)

		identities = os.listdir(self.identityFeatureDir)
		print "len(identities): ", len(identities)
		features = []
		for identity in identities:
			print "identity: ", identity
			feature = sio.loadmat(self.identityFeatureDir + identity)["identityFeature"]
			# .flatten()
			print "feature.shape: ", feature.shape
			features.extend(feature)

		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		features = np.asarray(features)
		print "len(identities): ", len(identities)
		print "features.shape: ", features.shape
		features = self.pca.fit_transform(features)
		print "features.shape: ", features.shape
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
		# np.savetxt('extracted_feature/' + self.name + 'PCAfeature.txt', features)


		identityFeatures = []
		preLabel = self.labelList[0]
		for index in range(len(self.labelList)):
			label = self.labelList[index]
			# identity = identities[index]
			feature = features[index]

			print feature[:10]
			if label == preLabel:
				identityFeatures.append(feature)
			else:
				print "preLabel: ", preLabel
				identityFeatures = np.asarray(identityFeatures)
				print "identityFeatures.shape: ", identityFeatures.shape
				identityFeatures = np.mean(identityFeatures, axis = 0)
				sio.savemat(self.PCAIdentityFeatureDir + preLabel, {"identityFeature": identityFeatures})
				print "save: ", self.PCAIdentityFeatureDir + preLabel
				preLabel = label
				identityFeatures = []

		identityFeatures = np.asarray(identityFeatures)
		print "identityFeatures.shape: ", identityFeatures.shape
		# identityFeatures = np.mean(identityFeatures, axis = 0)
		sio.savemat(self.PCAIdentityFeatureDir + preLabel, {"identityFeature": identityFeatures})
		print "save: ", self.PCAIdentityFeatureDir + preLabel
		preLabel = label

	def reducedIdentityDimTestData(self):
		chunk = 5000
		maxIter = 23
		batches = []
		for iter in range(maxIter + 1):
			print "iter_" + str(iter)
			print "loading features....."
			print 'extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt')
			print "batch.shape: ", batch.shape
			print "finish loading features....."
			batches.extend(batch)

		batches = np.asarray(batches)
		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		print "batches.shape: ", batches.shape
		batches = self.pca.fit_transform(batches)
		print "batches.shape: ", batches.shape
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"

		counter = 0
		for index in range(len(batches)):
			label = self.labelList[index]
			feature = batches[index]
			counter += 1
			sio.savemat("extracted_feature/C2test_featurePCA/" + label, {"identityFeature": feature})
			print label
			if counter % 100 == 0:
				print counter


	def writeToFile(self, content):
		with open('mxnetPredPCA.txt', 'a') as f:
			f.write(content)

	def testCosineSimilarity(self):
		with open('mxnetPredFull.txt', 'w') as f:
			f.write("")

		testIdentities = os.listdir(self.testDataPCAdir)
		# identities = os.listdir(self.totalPCAidentityFeatureDir)

		identities = os.listdir(self.totalIdentityFeatureDir)
		print identities[:10]

		predcontent = ""
		counter = 0
		try:
			for testIdentity in testIdentities:
				testIdentityFeature = sio.loadmat(self.testDataPCAdir + testIdentity)["identityFeature"]

				print "testIdentityFeature.shape: ", testIdentityFeature.shape
				scoreList = []
				for identity in identities:
					# identityFeature = sio.loadmat(self.totalPCAidentityFeatureDir + identity)["identityFeature"]
					identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"]
					# print identityFeature[:100]
					cosScore = 1 - float(spatial.distance.cosine(testIdentityFeature, identityFeature))
					# cosScore = cosine_similarity(feature, identityFeature)
					# print "identity: ", identity
					# print "cosScore: ", cosScore
					scoreList.append(cosScore)
				maxScore = max(scoreList)
				index = scoreList.index(maxScore)
				pred = identities[index]

				print "counter: ", counter
				print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
				print "label: ", testIdentity
				print "pred: ", pred
				print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
				predcontent += (pred + " " + str(maxScore) + "\n")
				counter += 1
				if counter % 100 == 0:
					self.writeToFile(predcontent)
					print "counter: ", counter
					predcontent = ""
		except Exception as e:
			print e
			self.writeToFile(predcontent)
			print "counter: ", counter
			content = ""
		self.writeToFile(predcontent)
		print "counter: ", counter
		predcontent = ""

	# def testCosineSimilarity(self):
	# 	with open('mxnetPredPCA.txt', 'w') as f:
	# 		f.write("")
	# 	chunk = 5000
	# 	maxIter = 23
	# 	testIdentities = os.listdir(self.testDataPCAdir)
	# 	identities = os.listdir(self.totalPCAidentityFeatureDir)
	# 	# identities = os.listdir(self.totalIdentityFeatureDir)
	# 	print identities[:10]
	#
	# 	predcontent = ""
	# 	counter = 0
	#
	# 	for iter in range(maxIter + 1):
	# 		print "loading features....."
	# 		print 'extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt'
	# 		batch = np.loadtxt('extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt')
	# 		print "finish loading features....."
	#
	# 		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
	# 		print "batch.shape: ", batch.shape
	# 	 	batch = self.pca.fit_transform(batch)
	# 		print "batch.shape: ", batch.shape
	# 		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
	# 		print "iter_" + str(iter)
	#
	# 		if iter == maxIter:
	# 			labelList = self.labelList[iter * chunk : ]
	# 		else:
	# 			labelList = self.labelList[iter * chunk : (iter + 1) * chunk]
	#
	# 		for index in range(len(labelList)):
	# 			try:
	# 				label = labelList[index]
	# 				feature = batch[index]
	# 				print "feature.shape: ", feature.shape
	# 				# feature = self.pca.fit_transform(feature)
	# 				# print "feature.shape: ", feature.shape
	# 				scoreList = []
	# 				for identity in identities:
	# 					identityFeature = sio.loadmat(self.totalPCAidentityFeatureDir + identity)["identityFeature"]
	# 					# identityFeature = sio.loadmat(self.totalIdentityFeatureDir + identity)["identityFeature"]
	# 					# print identityFeature[:100]
	# 					cosScore = 1 - float(spatial.distance.cosine(feature, identityFeature))
	# 					# cosScore = cosine_similarity(feature, identityFeature)
	# 					# print "identity: ", identity
	# 					# print "cosScore: ", cosScore
	# 					scoreList.append(cosScore)
	# 				maxScore = max(scoreList)
	# 				index = scoreList.index(maxScore)
	# 				pred = identities[index]
	# 				print "counter: ", counter
	# 				print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
	# 				print "label: ", label
	# 				print "pred: ", pred
	# 				print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
	# 				predcontent += pred + " " + str(maxScore)
	# 				counter += 1
	# 				if counter % 100 == 0:
	# 					self.writeToFile(predcontent)
	# 					print "counter: ", counter
	# 					content = ""
	# 			except Exception as e:
	# 				print e
	# 				self.writeToFile(predcontent)
	# 				print "counter: ", counter
	# 				content = ""
	# 		self.writeToFile(predcontent)
	# 		print "counter: ", counter
	# 		content = ""

	def run(self):
		# self.generateIdentityFeatures()
		self.reducedIdentityDimTrainData()
		# self.reducedIdentityDimTestData()
		# self.testCosineSimilarity()

if __name__ == '__main__':
	tcs = TestCosineSimilarity()
	tcs.run()
