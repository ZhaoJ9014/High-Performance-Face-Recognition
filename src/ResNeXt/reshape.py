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

path = '/media/samsung/learnnet_model_feature/learnnetNovelSet_1/'
files = os.listdir(path)
for file in files:
    feature = sio.loadmat(path + file)["identityFeature"]
    feature = feature.reshape((-1, 2048))
    print feature.shape
    break
