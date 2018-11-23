from FaceAligner_utils import face_alignment, FaceAligner, FaceResizer
import cv2
import os

source_root = './test'
dest_root = './test_aligned'
if not os.path.isdir(dest_root):
    os.mkdir(dest_root)

for subfolder in os.listdir(source_root):
    if subfolder != '.DS_Store':
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            if image_name != '.DS_Store':
                print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
                predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu').get_landmarks
                image = cv2.imread(os.path.join(source_root, subfolder, image_name))
                fa = FaceAligner(predictor)
                faceAligned = fa.align(image)
                ra = FaceResizer(desiredFaceSize=224)
                faceResized = ra.resize(faceAligned)
                cv2.imwrite(os.path.join(dest_root, subfolder, image_name), faceResized)