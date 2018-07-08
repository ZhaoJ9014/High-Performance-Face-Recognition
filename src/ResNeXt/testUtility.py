import cv2
import utility as ut

# generateFunc = ["original", "scale", "rotate", "translate", "scaleAndTranslate", "brightnessAndContrast"]
#
# def resize(originalImage, size = 224):
#
#     image = originalImage.copy()
#     # resize imgage to determined size maintaing the original ratio
#     w, h, _ = image.shape
#     if w >= h:
#         ratio = 224/float(w)
#         h = h * ratio
#         w = 224
#     else:
#         ratio = 224/float(h)
#         w = w * ratio
#         h = 224
#     image = cv2.resize(image, (int(h), int(w)))
#     return image

# img = cv2.imread("testImg/testImg.jpg", 1)
originalImg = cv2.imread("test.jpg", 1)
originalImg = cv2.resize(originalImg, (224, 224))

while 1:
    print originalImg.shape
    img = ut.randomCropImg(originalImg)
    # img = resize(originalImg)
    print img.shape
    # w, h, _ = img.shape
    # newImg = cv2.resize(newImg, (h, w))
    # newImg, x, y = ut.scale(img, [], [],  imSize = 224)
    # print newImg.shape
    # print newImg.shape

    # newImg, x, y = ut.mirror(img, [], [])
    # newImg, x, y = ut.contrastBrightess(img, [], [])
    # newImg, x, y = ut.rotate(img, [], [])
    # newImg, x, y = ut.translate(img, [], [])
    # newImg, x, y = ut.resize(img, [], [])


    cv2.imshow("originalImg", originalImg)
    cv2.imshow("img", img)
    # cv2.imshow("newImg", newImg)
    cv2.waitKey(0)
