from FaceAligner_utils import FaceResizer
import cv2
import os

source_root = '/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE'
dest_root = '/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE_Resized'
if not os.path.isdir(dest_root):
    os.mkdir(dest_root)

for subfolder in os.listdir(source_root):
    if subfolder == '.DS_Store' or subfolder == '._.DS_Store':
        continue
    else:
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            if image_name == '.DS_Store' or image_name == '._.DS_Store':
                continue
            else:
                print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
                image = cv2.imread(os.path.join(source_root, subfolder, image_name))
                ra = FaceResizer(desiredFaceSize=224)
                faceResized = ra.resize(image)
                cv2.imwrite(os.path.join(dest_root, subfolder, image_name), faceResized)
