import face_alignment
from helpers import FACIAL_LANDMARKS_68_IDXS
from helpers import FACIAL_LANDMARKS_5_IDXS
import numpy as np
import cv2

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35)):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye

    def align(self, image):
        shape = self.predictor(image)[0]

        self.FaceWidth = image.shape[1]
        self.FaceHeight = image.shape[0]

        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.FaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.FaceWidth * 0.5
        tY = self.FaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.FaceWidth, self.FaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

class FaceAligner_VGGFace2:
    def __init__(self, desiredLeftEye=(0.35, 0.35)):
        self.desiredLeftEye = desiredLeftEye

    def align(self, image, shape):
        shape = shape

        self.FaceWidth = image.shape[1]
        self.FaceHeight = image.shape[0]

        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.FaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.FaceWidth * 0.5
        tY = self.FaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.FaceWidth, self.FaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

class FaceResizer:
    def __init__(self, desiredFaceSize=224):
        self.desiredFaceSize = desiredFaceSize

    def resize(self, image):
        self.FaceWidth = image.shape[1]
        self.FaceHeight = image.shape[0]
        # apply resize with padding
        ratio = float(self.desiredFaceSize) / max(self.FaceHeight, self.FaceWidth)
        new_size = tuple([int(x * ratio) for x in (self.FaceHeight, self.FaceWidth)])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = self.desiredFaceSize - new_size[1]
        delta_h = self.desiredFaceSize - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        output = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # return the resized face
        return output


# predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu').get_landmarks
# image = cv2.imread('./faces/1198_0_861.jpg')
# fa = FaceAligner(predictor)
# faceAligned = fa.align(image)
# ra = FaceResizer(desiredFaceSize=224)
# faceResized = ra.resize(faceAligned)
# cv2.imwrite("foo/" + "1.png", faceResized)