import scipy.io as sio
import pickle
import numpy as np
import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from scipy import spatial


class TestCosineSimilarity(object):
	def __init__(self):
		# self.name = "C2test_feature"
		self.name = "MSchallenge2Base"
		reducedDim = 512
		self.pca = PCA(n_components = reducedDim)
		self.identityFeatureDir = "extracted_feature/MSchallenge2BaseIdentityMeanFeature/"
		self.PCAIdentityFeatureDir = "extracted_feature/MSchallenge2BaseIdentityMeanFeaturePCA/"
		self.labelList = pickle.load(open(self.name + "LabelList.p", "rb"))
		print len(self.labelList)

		self.path = "extracted_feature/" + self.name + "IdentityMeanFeature/"
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
			batch = np.loadtxt('extracted_feature/' + self.name + '_feature/' + self.name + '_feature_batch' + str(iter) + '.txt')
			print "iter_" + str(iter), " ", batch.shape

			if iter == maxIter:
				labelList = self.labelList[iter : ]
			else:
				labelList = self.labelList[iter : (iter + 1) * chunk]

			if len(preFeatures) != 0:
				features = preFeatures
			else:
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
					features = np.asarray(features)
					identityFeature = np.mean(features, axis = 0)
					print "identityFeature.shape: ", identityFeature.shape
					sio.savemat(self.path + preLabel, {"identityFeature": identityFeature})
					preLabel = label
					features = []
					features.append(feature)
			if len(features) != 0 and iter != maxIter:
				preFeatures = features
			else:
				features = np.asarray(features)
				identityFeature = np.mean(features, axis = 0)
				print "identityFeature.shape: ", identityFeature.shape
				sio.savemat(self.path + preLabel, {"identityFeature": identityFeature})

	def reducedIdentityDim(self):
		if not os.path.isdir(self.PCAIdentityFeatureDir):
			os.mkdir(self.PCAIdentityFeatureDir)

		identities = os.listdir(self.identityFeatureDir)
		for identity in identities:
			feature = sio.loadmat(self.identityFeatureDir + identity)["identityFeature"]
			feature = self.pca.fit_transform(feature)
			sio.savemat(self.PCAIdentityFeatureDir + identity, {"identityFeature": identityFeature})

	def writeToFile(self, content):
		with open('mxnetPred.txt', 'a') as f:
			f.write(content)

	def testCosineSimilarity(self):

		chunk = 5000
		maxIter = 23
		scoreList = []
		identities = os.listdir(self.PCAIdentityFeatureDir)
		predcontent = ""
		counter = 0
		for iter in range(maxIter + 1):
			print "loading features....."
			print 'extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt'
			batch = np.loadtxt('extracted_feature/C2test_feature/' + self.name + '_feature_batch' + str(iter) + '.txt')
			print "finish loading features....."
			print "iter_" + str(iter), " ", batch.shape

			if iter == maxIter:
				labelList = self.labelList[iter : ]
			else:
				labelList = self.labelList[iter : (iter + 1) * chunk]

			for index in range(len(labelList)):
				label = labelList[index]
				print "label: ", label
				feature = featureList[index]
				print "feature.shape: ", feature.shape
				feature = self.pca.fit_transform(feature)
				print "feature.shape: ", feature.shape

				for identity in identities:
					identityFeature = sio.loadmat(self.PCAIdentityFeatureDir + identity)["identityFeature"]
					cosScore = 1 - spatial.distance.cosine(feature, identityFeature)
					print "cosScore: ", cosScore
					scoreList.append(cosScore)
				maxScore = max(scoreList)
				index = scoreList.index(maxScore)
				pred = identities[index]
				print "pred: ", pred
				predcontent += pred + " " + str(maxScore)
				counter += 1
				if counter % 100 == 0:
					self.writeToFile(predcontent)
					print "counter: ", counter
					content = ""

	def run(self):
		# self.reducedDim()
		self.generateIdentityFeatures()

if __name__ == '__main__':
	tcs = TestCosineSimilarity()
	tcs.run()
