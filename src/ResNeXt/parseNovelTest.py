def writeToFile(content):
	name = "novelTestList"
	with open(name + '.txt', 'a') as f:
		f.write(content)

writeToFile("")

f = open('noveltest.txt', 'r')

lines = f.readlines()

counter = 0
content = ""
for line in lines:
	mid = line.split(" ")[0].replace("\\", "/").split(" ")[0].replace("I:/lowshot/dev/img2/NovelSet/", "").replace("\r\n", "").split("/")[0]
	path = line.split(" ")[0].replace("\\", "/").split(" ")[0].replace("I:/lowshot/dev/img2/NovelSet/", "").replace("\r\n", "")
	path = "/media/zhaojian/6TB/data/img2/NovelSet/" + path
	content += path + "\t" +  mid + "\n"
	counter += 1
	if counter % 100 == 0:
		print "counter: ", counter
		writeToFile(content)
		content = ""
writeToFile(content)
content = ""
