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

features = []
identityFeatureDir = ""
identities = os.listdir(identityFeatureDir)
for identity in identities:
	feature = sio.loadmat(identityFeatureDir + identity)["identityFeature"].flatten()
	print "feature.shape: ", feature.shape
	features.append(feature)
	labelList.append(identity)

features = np.asarray(features)
print "len(labelList): ", len(labelList)
print "features.shape: ", features.shape
			sio.savemat(self.PCAIdentityFeatureDir + label, {"identityFeature": feature})
