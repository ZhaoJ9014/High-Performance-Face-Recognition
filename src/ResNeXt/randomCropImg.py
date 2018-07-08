import random, os, time
from PIL import Image
import Image, ImageOps

# def padding(old_im):
#     # old_im = Image.open('someimage.jpg')
#     old_size = old_im.size
#
#     new_size = (256, 256)
#     new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
#     new_im.paste(old_im, ((new_size[0]-old_size[0])/2,
#                           (new_size[1]-old_size[1])/2))


INPATH = r".../images"
OUTPATH = r".../tiles"

dx = dy = 256
tilesPerImage = 100

files = os.listdir(INPATH)
numOfImages = len(files)

t = time.time()
for file in files:
   with Image.open(os.path.join(INPATH, file)) as im:

     for i in range(1, tilesPerImage+1):
       newname = file.replace('.', '_{:03d}.'.format(i))
       w, h = im.size
       x = random.randint(0, w-dx-1)
       y = random.randint(0, h-dy-1)
       print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
       im.crop((x,y, x+dx, y+dy))\
         .save(os.path.join(OUTPATH, newname))

t = time.time()-t
print("Done {} images in {:.2f}s".format(numOfImages, t))
print("({:.1f} images per second)".format(numOfImages/t))
print("({:.1f} tiles per second)".format(tilesPerImage*numOfImages/t))
