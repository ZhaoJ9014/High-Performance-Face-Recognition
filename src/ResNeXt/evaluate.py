import scipy.io as sio
import pickle
from time import sleep
# name = "C2testResized"
name = "C2test"

# labelList = pickle.load(open(name + "LabelListPCA.p", "rb"))
labelList = pickle.load(open(name + "LabelList.p", "rb"))
print len(labelList)

# name = "mxnetPredPCA1024_new.txt"
# name = "mxnetPred_not_pca2017-06-12-05-34.txt"
# name = "mxnetPredPCA5122017-06-12-05-46.txt"
# name = "mxnetPred2048_pca_test2017-06-12-14-39.mat"
# name = "mxnetPredPCA512_pca_test2017-06-12-17-39.mat"
# name = "mxnetPredPCA2048_pca_test2017-06-13-01-15.mat"
# name = "mxnetPred2017-06-13-01-53.mat"
# name = "mxnetPredPCA2048_pca_test2017-06-13-11-44.mat"
# name = "mxnetPred2017-06-13-12-56.mat"
# name = "mxnetPred2017-06-13-13-00.mat"
# name = "mxnetPred2017-06-13-15-15.mat"
# name = "mxnetPred2017-06-13-15-21.mat"
# name = "mxnetPred512_pca_test2017-06-12-17-36.mat"
# name = "mxnetPred512_pca_test2017-06-12-17-38.mat"
name = "mxnetPred_pca_test2017-06-13-16-57.mat"
# name = "mxnetPred_pca_test2017-06-13-16-58.mat"

lines = sio.loadmat(name)['data']

# with open(name, 'r') as f:
# 	lines = f.readlines()

print "len(lines): ", len(lines)

# print "sleep for 5"
# sleep(5)

result = []
for index in range(len(lines)):
    line = lines[index]
    label = labelList[index]
    # print "label: ", label
    # print "line: ", line
    label = label.replace(".mat", "")
    pred = line.split(" ")[0].replace(".mat", "")
    # print "pred: ", pred
    # print "label: ", label
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
