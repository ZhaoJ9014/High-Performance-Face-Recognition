import pickle
import numpy as np
name = "DevNovelSet"
labelList = pickle.load(open('extracted_feature/' + name + "LabelList.p", "rb"))
print len(labelList)
print labelList[0]
labelList = np.asarray(labelList)
# np.savetxt('extracted_feature/' + name + "LabelList.txt", labelList)


file = open('extracted_feature/' + name + "LabelList.txt", 'w')
for item in labelList:
    file.write(item + "\n")
