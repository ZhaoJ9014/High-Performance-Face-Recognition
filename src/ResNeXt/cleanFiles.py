import sys
import os
from os import listdir
from os.path import join

# clean deirvative images for face data
dir = "/media/zhaojian/6TB/data/MS-Celeb-1M/NovelSet_1_mirror/"
# dir = "/home/james/MS_GAN/data/10train/"
# dir = "/Users/zhecanwang/Project/MS_GAN/data/10train/"
folders =os.listdir(dir)
for folder in folders:
    if ".DS_Store" not in folder:
        test=os.listdir(dir + folder)
        for item in test:
            if item.endswith("_2.jpg") or item.endswith("_1.jpg") or item.endswith("_0.jpg") or item.endswith("_3.jpg") or item.endswith("_4.jpg"):
                pass
            else:
                os.remove(join(dir + folder, item))


# clean model weights
# folder = "06152017_each_10000_black_white"
# dir = "/home/james/MS_GAN/models/generator/" + folder + "/"
# files =os.listdir(dir)
# for file in files:
#     if ".DS_Store" not in file:
#         print file
#         suffix = int(file[len("params_generator_epoch_"):].split(".")[0])
#         if suffix < 150:
#             os.remove(join(dir, file))
