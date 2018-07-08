
import numpy as np
import pickle

outputDir = "extracted_feature/"
# def writeToFile(content):
# 	with open(outputDir + 'dummy_labels.txt', 'a') as f:
# 	# with open('dummy_labels.txt', 'a') as f:
# 		f.write(content)


# path = "extracted_feature/"
# name = "challenge2"

path = "/home/zhaojian/DEEP/JK_GoogleNet_BN/list/"
name = "JK_BaseSet_first_part"

with open(path + name + '.txt', 'r') as f:
	lines = f.readlines()
print len(lines)

# labels = []
# counter = 0
# for line in lines:
# 	split = line.split("\t")
# 	label = split[1].replace("\n", "")
# 	labels.append(label)
# 	counter += 1
# 	# if counter >= 130:
# 	# 	break
# 	if counter%1000  == 0:
# 		print "counter: ", counter


labels = []
counter = 0
for line in lines:
	label = line.split(" ")[0].split("/")[-2]
	labels.append(label)
	counter += 1
	if counter%1000  == 0:
		print "counter: ", counter


print "len(labels): ", len(labels)

pickle.dump( labels, open( outputDir + name + "LabelList.p", "wb" ) )
