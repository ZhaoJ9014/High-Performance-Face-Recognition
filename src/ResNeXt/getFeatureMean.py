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

identityFeatureDir = "extracted_feature/BaseSetIdentityFeature/"
identities = os.listdir(identityFeatureDir)
for identity in identities:
	identityFeature = sio.loadmat(identityFeatureDir + identity)["identityFeature"]
	print "feature.shape: ", identityFeature.shape

	identityFeature = np.mean(identityFeature, axis = 0)
	print "feature.shape: ", identityFeature.shape

	sio.savemat("extracted_feature/BaseSetIdentityFeatureMean/" + identity, {"identityFeature": identityFeature})
