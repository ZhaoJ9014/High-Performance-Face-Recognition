with open('lowshotImg_cropped5_224.txt', 'r') as f:
	lines = f.readlines()

# with open('MStrain.txt', 'r') as f:
# 	lines = f.readlines()

idDict = {}

for line in lines:
	ID = line.split("\t")[1]
	idDict[ID] = ""

print len(idDict.keys())
