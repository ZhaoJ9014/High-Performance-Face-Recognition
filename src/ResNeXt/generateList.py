import os

# /media/zhaojian/6TB/data/MS-Celeb-1M/challenge2
name = "challenge2"
# tag = "Dev"
tag = ""

def writeToFile(content):
	# name = "DevNovelSet_randomCrop"
	with open(tag + name + '.txt', 'a') as f:
		f.write(content)

writeToFile("")

counter = 0
inputDir = "/media/zhaojian/6TB/data/MS-Celeb-1M/" + name + "/"
# inputDir = "/media/zhaojian/6TB/data/img2/" + name + "/"

files = os.listdir(inputDir)
print "len(files): ", len(files)
content = ""
# for file in files:
# 	if "m." in file:
# 		pics = os.listdir(inputDir + file)
# 		for pic in pics:
# 			path = (inputDir + file + "/" + pic)
# 			content += path + "\t" + file + "\n"
# 			counter += 1
# 			if counter % 100 == 0:
# 				print "counter: ", counter
# 				writeToFile(content)
# 				content = ""
for file in files:
	path = (inputDir + file)
	content += path + "\t" + file + "\n"
	counter += 1
	if counter % 100 == 0:
		print "counter: ", counter
		writeToFile(content)
		content = ""
writeToFile(content)
content = ""
