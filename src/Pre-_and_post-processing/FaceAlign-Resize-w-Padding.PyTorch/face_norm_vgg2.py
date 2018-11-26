from FaceAligner_utils import face_alignment, FaceAligner_VGGFace2, FaceResizer
import cv2
import os
import numpy as np

root = '/home/zhaojian/zhaojian/DATA/VGGFACE2_Cleandata'
source_root = '/home/zhaojian/zhaojian/DATA/VGGFACE2_Cleandata/train'
dest_root = '/home/zhaojian/zhaojian/DATA/VGGFACE2_Aligned'
anno_filename = 'VGGFACE2_cleandata_5pts.txt'
if not os.path.isdir(dest_root):
    os.mkdir(dest_root)

def read_anno(anno_filename):
    paths = []
    annos = []
    with open(anno_filename, 'r') as f:
        for line in f.readlines():
            path = line.strip().split()[0]
            anno = line.strip().split()[1:]
            paths.append(path)
            annos.append(anno)
    return paths, np.array(annos)

paths, annos = read_anno(os.path.join(root, anno_filename))
for path, anno in zip(paths, annos):
    subfolder = path.split('/')[0]
    if not os.path.exists(os.path.join(dest_root, subfolder)):
        os.mkdir(os.path.join(dest_root, subfolder))
    image_name = path.split('/')[1]
    print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
    image = cv2.imread(os.path.join(source_root, subfolder, image_name))
    fa = FaceAligner_VGGFace2()
    faceAligned = fa.align(image, anno)
    ra = FaceResizer(desiredFaceSize=224)
    faceResized = ra.resize(faceAligned)
    cv2.imwrite(os.path.join(dest_root, subfolder, image_name), faceResized)