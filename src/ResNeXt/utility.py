import math
import cv2
import numpy as np
from PIL import Image, ImageChops
from pylab import array, uint8
import PIL
import random

def mirror(image, X, Y, h = None, w = None):
	image = np.fliplr(image)
	# X = w - X
	# X = np.asarray(X)
	# Y = np.asarray(Y)
	# return image, X.astype(int), Y.astype(int)
	return image, X, Y

def contrastBrightess(image, X, Y):
	contrast = np.random.uniform(0.5, 3)
	brightness = np.random.uniform(-50, 50)
	# contrast = 2
	# brightness = 50

	maxIntensity = 255.0 # depends on dtype of image data
	phi = 1
	theta = 1
	image = ((maxIntensity/phi)*(image/(maxIntensity/theta))**contrast) + brightness
	image = np.asarray(image)
	top_index = np.where(image > 255)
	bottom_index = np.where(image < 0)
	image[top_index] = 255
	image[bottom_index] = 0
	image = array(image,dtype=uint8)
	X = np.asarray(X)
	Y = np.asarray(Y)
	return image, X, Y


def rotate(image, X, Y, h = None, w = None, counter = 0, random = None):
   # if counter > 2:
   #     return None, None, None
	originalImage = image.copy()

	if h == None:
		(h, w, _) = image.shape

	degree = np.random.uniform(-30, 30)

	M = cv2.getRotationMatrix2D((w/2, h/2),degree,1)

	newX = []
	newY = []
	for x, y in zip(X, Y):
		newX.append(M[0, 0] * x + M[0, 1] * y + M[0, 2])
		newY.append(M[1, 0] * x + M[1, 1] * y + M[1, 2])

	image = cv2.warpAffine(image,M,(w, h))

	#if np.min(newX) < 0 or np.min(newY) < 0 or np.max(newX) > w or np.max(newY) > h:
	#    image, newX, newY = rotate(originalImage, X, Y, counter + 1)

	newX = np.asarray(newX)
	newY = np.asarray(newY)
	# print "image: ", type(image)
	return image, newX, newY


def resize(originalImage, size = 224):

	image = originalImage.copy()
	# resize imgage to determined size maintaing the original ratio
	w, h, _ = image.shape
	if w >= h:
		ratio = 224/float(w)
		h = h * ratio
		w = 224
	else:
		ratio = 224/float(h)
		w = w * ratio
		h = 224
	image = cv2.resize(image, (int(h), int(w)))
	return image

# def resize(originalImage, X, Y, xMaxBound = None, yMaxBound = None, random = False, size = None, debug = False):
#
#     image = originalImage.copy()
#     # resize imgage to determined size maintaing the original ratio
#
#     if yMaxBound == None:
#         (yMaxBound, xMaxBound, _) = image.shape
#
#     if debug:
#         print "initial (yMaxBound, xMaxBound, _): ", (yMaxBound, xMaxBound, _)
#     newX = [x/float(xMaxBound) for x in X]
#     newY = [y/float(yMaxBound) for y in Y]
#
#
#     if random:
#         ratio = np.random.uniform(0.7, 1.5)
#         size = (int(xMaxBound*ratio), int(yMaxBound*ratio))
#         print "ratio: ", ratio
#
#     if debug:
#         print "determined size: ", size
#     image = Image.fromarray(np.uint8(image))
#     # image.thumbnail(size, Image.ANTIALIAS)
#
#     if size[0] > size[1]:
#         basewidth = size[0]
#         wpercent = (basewidth/float(image.size[0]))
#         hsize = int((float(image.size[1])*float(wpercent)))
#         image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
#     else:
#         hsize = size[1]
#         hpercent = (hsize/float(image.size[1]))
#         basewidth = int((float(image.size[0])*float(hpercent)))
#         image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
#
#     image_size = image.size
#
#     if debug:
#         image.show()
#         print "after resize image.size: ", image.size
#
#     # if random:
#     #     (newXMaxBound, newYMaxBound) = size
#     # else:
#     (newXMaxBound, newYMaxBound) = image.size
#
#     newX = [x*float(newXMaxBound) for x in newX]
#     newY = [y*float(newYMaxBound) for y in newY]
#
#     thumb = image.crop( (0, 0, size[0], size[1]) )
#     image = np.asarray(thumb)
#
#     if debug:
#         print "size: ", size
#         print "type(np.asarray(thumb).shape): ", np.asarray(thumb).shape
#         cv2.imshow("img", image)
#         cv2.waitKey(0)
#     # offset_y = (size[0] - image_size[1]) / 2
#     # offset_x = (size[1] - image_size[0]) / 2
#     # print "offset_x: ", offset_x
#     # print "offset_y: ", offset_y
#
#     # if offset_x <= 0:
#     #     newX = [x + offset_x for x in newX]
#     #     newY = [y + offset_y for y in newY]
#     # else:
#     #     newX = [x - offset_x for x in newX]
#     #     newY = [y - offset_y for y in newY]
#
#     # thumb = ImageChops.offset(thumb, offset_x, offset_y)
#     # image = np.asarray(thumb)
#
#     if random:
#         # newImg = np.zeros_like(originalImage)
#         print "newXMaxBound, newYMaxBound: ", newXMaxBound, newYMaxBound
#         newImg = np.zeros((newXMaxBound, newYMaxBound, 3))
#         print "yMaxBound: ", yMaxBound
#         print "xMaxBound: ", xMaxBound
#         print "image_size: ", image_size
#         print "int((image_size[1] - yMaxBound) / 2): ", int((image_size[1] - yMaxBound) / 2)
#
#         # print "yMaxBound - image_size[1]: ", yMaxBound - image_size[1]
#         # if ratio <= 1:
#         #     offset_y = int((yMaxBound - hsize) / 2)
#         #     offset_x = int((xMaxBound - basewidth) / 2)
#         #     other_offset_y = -offset_y if image_size[1] % 2 == 0 else -(offset_y + 1)
#         #     other_offset_x = -offset_x if image_size[0] % 2 == 0 else -(offset_x + 1)
#
#         # else:
#         offset_y = int((image_size[1] - hsize) / 2)
#         offset_x = int((image_size[0] - basewidth) / 2)
#
#         other_offset_y = -offset_y if image_size[1] % 2 == 0 else -(offset_y - 1)
#         other_offset_x = -offset_x if image_size[0] % 2 == 0 else -(offset_x - 1)
#         other_offset_x = other_offset_x + image_size[1]
#         other_offset_y = other_offset_y + image_size[0]
#
#         newX = [x + offset_x for x in newX]
#         newY = [y + offset_y for y in newY]
#
#         print "offset_y:other_offset_y: ", offset_y, other_offset_y
#         print "offset_x:other_offset_x: ", offset_x, other_offset_x
#         print " newImg.shape: ", newImg.shape
#         print "newImg[offset_y:other_offset_y, offset_x:other_offset_x]: ", newImg[offset_y:other_offset_y, offset_x:other_offset_x].shape
#         print "image.shape", image.shape
#         # newImg[offset_y:other_offset_y, offset_x:other_offset_x] = image
#         newImg[offset_y:other_offset_y, offset_x:other_offset_x] = image
#         image = newImg
#
#     newX = np.asarray(newX)
#     newY = np.asarray(newY)
#
#     return image, newX.astype(int), newY.astype(int)

def scale(image, X, Y, imSize = None):
	originalImage = image
	# resize imgage to determined size maintaing the original ratio
	(yMaxBound, xMaxBound, _) = image.shape

	if imSize == None:
		ratio = np.random.uniform(0.7, 1.5)
		imSize = int(max(xMaxBound, yMaxBound) * ratio)
		size = (imSize, imSize)
	else:
		size = (imSize, imSize)

	# print "size: ", size

	newX = [x/float(xMaxBound) for x in X]
	newY = [y/float(yMaxBound) for y in Y]

	image = Image.fromarray(np.uint8(image))
	image.thumbnail(size, Image.ANTIALIAS)
	image_size = image.size

	(newXMaxBound, newYMaxBound) = image.size

	newX = [x*float(newXMaxBound) for x in newX]
	newY = [y*float(newYMaxBound) for y in newY]

	thumb = image.crop( (0, 0, size[0], size[1]) )
	image = np.asarray(thumb)

	offset_y = (size[0] - image_size[1]) / 2
	offset_x = (size[1] - image_size[0]) / 2

	newX = [x + offset_x for x in newX]
	newY = [y + offset_y for y in newY]

	thumb = ImageChops.offset(thumb, offset_x, offset_y)

	image = np.asarray(thumb)

	newX = np.asarray(newX)
	newY = np.asarray(newY)

	return image, newX.astype(int), newY.astype(int)

def randomCropImg(img):
	# img = resize(img)
	# newImg, x, y = scale(img, [], [],  imSize = 256)
	img = cv2.resize(img, (256, 256))
	cropRange = 256 - 224
	leftMost_x = random.randint(0, cropRange)
	leftMost_y = random.randint(0, cropRange)

	crop_img = img[leftMost_x : leftMost_x + 224, leftMost_y : leftMost_y + 224] # Crop from x, y, w, h -> 100, 200, 300, 400
	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
	return crop_img

def translate(image, X, Y, w = None , h = None, counter = 0):
	X, Y = np.asarray(X), np.asarray(Y)
	# if counter > 10:
	#     return None, None, None
	originalImage = image.copy()
	(h, w, _) = image.shape
	xTransRange, yTransRange = np.random.randint(0, w/3), np.random.randint(0, h/3)

	newImg = np.zeros_like(image)
	newImg[yTransRange:, xTransRange:] = image[:int(h - yTransRange), :int(w - xTransRange)]

	newX = X + xTransRange
	newY = Y + yTransRange

	image = newImg

	# if np.min(newX) < 0 or np.min(newY) < 0 or np.max(newX) > w or np.max(newY) > h:
	#     image, newX, newY = translate(originalImage, X, Y, counter + 1)

	newX = np.asarray(newX)
	newY = np.asarray(newY)
	return image, newX, newY

def packLandmarks(X, Y):
	pts = zip(X, Y)
	landmarks = []
	for p in pts:
		landmarks.append(list(p))
	return landmarks

def unpackLandmarks(array, imSize):
	x = []
	y = []
	for i in range(0, len(array)):
		x.append((array[i][0] + 0.5) * imSize)
		y.append((array[i][1] + 0.5) * imSize)
	return x, y

def deNormalize(array, imSize):
	if isinstance(array, list):
		array = list(array)
		newArray = []
		for i in range(len(array)):
			newArray.append((array[i] + 0.5) * float(imSize))
		return newArray
	else:
		return (array+ 0.5) * float(imSize)


def normalize(array, imSize):
	if isinstance(array, list):
		array = list(array)
		newArray = []
		for i in range(len(array)):
			newArray.append((array[i]/float(imSize)) - 0.5)
		return newArray
	else:
		return (array/float(imSize)) - 0.5


def plotTarget(image, labels, imSize = None, ifSquareOnly = False, ifGreen = False):
	img = np.copy(image)
	# assert len(labels) == 7

	# try:
	# (w, h, _) = (imSize, imSize, 0)
	if ifSquareOnly:
		xMean = labels[0]
		yMean = labels[1]
		edge = labels[2]
	else:
		for i in range(0, 6, 2):
			# if int(labels[i]) <= 128 and int(labels[i + 1]) <= 128
			#     if int(labels[i]) >= 0 and int(labels[i + 1]) >= 0
			cv2.circle(img,(int(labels[i]), int(labels[i + 1])), 2, (0,0,255), -1)
		xMean = labels[4]
		yMean = labels[5]
		edge = labels[6]
	if ifGreen:
		cv2.rectangle(img,(int(xMean - edge/2.0), int(yMean - edge/2.0)),(int(xMean + edge/2.0),
			int(yMean + edge/2.0)),(0, 255, 0), 3)
	else:
		cv2.rectangle(img,(int(xMean - edge/2.0), int(yMean - edge/2.0)),(int(xMean + edge/2.0),
			int(yMean + edge/2.0)),(255, 0, 0), 3)
	# except Exception as e:
	#     print e
	#     print "plotTarget labels: ", labels
	return img

def plotLandmarks(image, X, Y, imSize = None, name = None, ifRescale = False, ifReturn = False, circleSize = 2):
	# plot landmarks on original image
	# img = np.copy(image)
	img = image.copy()
	assert len(X) == len(Y)
	# print "X: :::::::::::::::",
	# print X[:10]
	for index in range(len(X)):
		if ifRescale:
			(w, h, _) = img.shape
			# (w, h, _) = (128, 128, 0)
			cv2.circle(img,(int((X[index] + 0.5) * imSize), int((Y[index] + 0.5) * imSize)), circleSize, (0,0,255), -1)
		else:
			cv2.circle(img,(int(X[index]), int(Y[index])), circleSize, (0,0,255), -1)
	if ifReturn:
		return img
	else:
		cv2.imshow(name,img)


def test():
	dataDir = "./data/ibug/"
	# Load an color image in grayscale
	picName = "image_008_1.jpg"
	ptsName = "image_008_1.pts"


	img = cv2.imread(dataDir + picName,1)
	cv2.imshow("original", img)

	(ymax, xmax, _) = img.shape

	file = open(dataDir + ptsName, 'r')
	initDataCounter = 0
	X = []
	Y = []

	for point in file:
		if initDataCounter > 2:
			if "{" not in point and "}" not in point:
				strPoints = point.split(" ")
				x = int(float(strPoints[0]))
				y = int(float(strPoints[1]))
				X.append(x)
				Y.append(y)
		else:
			initDataCounter += 1


	X = np.asarray(X)
	Y = np.asarray(Y)
	print X.shape
	print Y.shape
	# cleanImg = np.copy(img)
	cleanImg = img.copy()

	plotLandmarks(img, X, Y, "img")
	while True:

		mirImage, newX, newY = mirrorImage(cleanImg, X, Y)
		mirImage = mirImage.copy()
		cv2.imshow("mirrorClean", mirImage)
		plotLandmarks(mirImage, newX, newY, "mirror")
		print newX.shape
		print newY.shape

		resizeImage, newX, newY = resize(cleanImg, X, Y)

		cv2.imshow("resizeClean", resizeImage)
		cleanResizeImage = np.copy(resizeImage)
		plotLandmarks(resizeImage, newX, newY, "resize")

		resizeImage1, newX, newY = resize(cleanResizeImage, newX, newY, random = True)
		cv2.imshow("resizeClean1", resizeImage1)
		plotLandmarks(resizeImage1, newX, newY, "resize1")

		rotateImage, newX, newY = rotate(cleanImg, X, Y)
		cv2.imshow("rotateClean", rotateImage)
		plotLandmarks(rotateImage, newX, newY, "rotate")

		cbImage, newX, newY = contrastBrightess(cleanImg, X, Y)
		cv2.imshow("contrastBrightnessClean", cbImage)
		plotLandmarks(cbImage, newX, newY, "contrastBrightness")

		transImage, newX, newY = translateImage(cleanImg, X, Y)
		cv2.imshow("transClean", transImage)
		plotLandmarks(transImage, newX, newY, "trans")

		cv2.waitKey(1000)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	test()
