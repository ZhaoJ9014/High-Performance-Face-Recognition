import os
import scipy.io as sio
import numpy as np
import pickle

# reduced_features = sio.loadmat('extracted_feature/C2testIdentityFeatureIdentityFeatureReduced.mat')['Fea_Sub_Orig_NovelSet']
reduced_features = sio.loadmat('extracted_feature/Challenge2Feature5IdentityFeatureReduced.mat')['Fea_Sub_Orig_NovelSet']
print type(reduced_features)
print reduced_features.shape

label_list = pickle.load(open("extracted_feature/Challenge2Feature5IdentityFeature.p", "rb"))
# label_list = pickle.load(open("C2testLabelList.p", "rb"))

output_dir = "extracted_feature/totalIdentityFeaturePCA_matlab/"

if len(reduced_features) != len(label_list):
	raise "len(features) != len(collectMatList)"
features = []
for index in range(len(label_list)):
	label = label_list[index]
	feature = reduced_features[index]
	print "feature.shape: ", feature.shape
	sio.savemat(output_dir + label, {"identityFeature": feature})
	# features.append(feature)

# features = np.asarray(features)
# sio.savemat('extracted_feature/C2test_featurePCA_matlab', {"identityFeature": features})
