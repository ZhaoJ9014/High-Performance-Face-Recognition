# /Volumes/usb1/ms2/
#
# /media/usbstick/MS_challenge1_train_data
#
#
def writeToFile(content):
	with open('MStrainFloat2.lst', 'a') as f:
		f.write(content)


with open('MStrainFloat.lst', 'r') as f:
	lines = f.readlines()

counter = 0
content = ""
for line in lines:
	line = line.replace("/Volumes/usb1/ms2/", "/media/usbstick/MS_challenge1_train_data/")
	line = line.replace('.jpg', '.png')
	content += line
	counter += 1
	if counter % 100 == 0:
		writeToFile(content)
		content = ""
		print counter
