# def writeToFile(self, content):
#   with open('mxnetPredPCA.txt', 'a') as f:
#     f.write(content)

with open('mxnetPredPCA.txt', 'r') as f:
	lines = f.readlines()
# count = 0
for line in lines:
	# line = line.split(" ")
	print line
	# break
#     print line[:10]
#     print len(line)
#     count += len(line)
# print count
#     # i = 0
#     # while (i + 2 < len(line) -1):
#     #     pred = line[i]
#     #     score = line[i + 2]
#     #     i += 2
#     #     print pred
#     #     print score
#     #     break
#     # break
# import pickle
# name = "C2test"
# labelList = pickle.load(open(name + "LabelList.p", "rb"))
# print len(labelList)
