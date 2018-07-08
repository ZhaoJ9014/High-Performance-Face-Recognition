# /Volumes/usb1/ms2/
#
# /media/usbstick/MS_challenge1_train_data
#
#

import pickle
#
# def writeToFile(content):
# 	with open('/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature//MS_dev_set/MS_dev_set_feature.txt', 'a') as f:
#
# 		f.write(content)
#
#
# with open('/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/DevNovelSet_feature_batch/DevNovelSet_Feature.txt', 'r') as f:
# 	lines1 = f.readlines()
# with open('/media/zhaojian/6TB/project/Resnext_MS_model_Simple/extracted_feature/DevBaseSet_feature_batch/DevBaseSet_Feature.txt', 'r') as f:
# 	lines2 = f.readlines()
#
# counter = 0
# content = ""
# for line in lines1:
# 	content += line
# 	counter += 1
# 	if counter % 100 == 0:
# 		writeToFile(content)
# 		content = ""
# 		print counter
#
# writeToFile(content)
# content = ""
# print counter
#
# for line in lines2:
# 	content += line
# 	counter += 1
# 	if counter % 100 == 0:
# 		writeToFile(content)
# 		content = ""
# 		print counter
#
# writeToFile(content)
# content = ""
# print counter
#
#
# name = "DevNovelSet"
# labelList1 = pickle.load(open("extracted_feature/" + name + "LabelList.p", "rb"))
#
# name = "DevBaseSet"
# labelList2 = pickle.load(open("extracted_feature/" + name + "LabelList.p", "rb"))
#
# counter = 0
# content = []
# for line in labelList1:
# 	content.append(line)
# 	counter += 1
# 	if counter % 100 == 0:
# 		print counter
#
# for line in labelList2:
# 	content.append(line)
# 	counter += 1
# 	if counter % 100 == 0:
# 		print counter
#
# pickle.dump( content, open( "extracted_feature/MS_dev_set_labelList.p", "wb" ) )
testLabelList = pickle.load(open("extracted_feature/MS_dev_set_labelList.p", "rb"))
print len(testLabelList)
