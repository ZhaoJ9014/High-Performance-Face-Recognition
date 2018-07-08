import numpy as np
features = np.loadtxt('extracted_feature/MSchallenge2BaseFeature.txt')
print "type(features): ", type(features)
print "features.shape: ", features.shape
import os
import cv2
def writeToFile(content):
	with open('MSchallenge2Base_tab1.txt', 'a') as f:
		f.write(content)


with open('MSchallenge2Base_tab.txt', 'r') as f:
	lines = f.readlines()

counter = 0
content = ""
for line in lines:
    path = line.split("\t")[0]
    path = path[len("/home/james/MS-Celeb-1M/"):]
    img = cv2.imread(path)
    if img != None:
    	content += line
    	counter += 1
	if counter % 100 == 0:
		writeToFile(content)
		content = ""
		print counter
writeToFile(content)
content = ""
print counter
