#mklist_path=/home/xiong/mxnet/tools
#python $mklist_path/make_list.py --recursive 1 ./Vgg_lmk vgg_face --train_ratio 0.9 --test_ratio 0.05
#python make_list.py --recursive 1 ./train sf1 --train_ratio 0.9
#make sure conda install opencv
#im2rec_path=/home/tairuic/Downloads/mxnet/bin/im2rec
# im2rec_path=/home/xiong/mxnet/tools
im2rec_path='/home/zhaojian/mxnet-master/tools'

SHAPE=224
QUALITY=90
#python $im2rec_path/im2rec.py train_${SHAPE}_sq.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
#python $im2rec_path/im2rec.py test_sq.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
#python $im2rec_path/im2rec.py Probe_sq.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
# python $im2rec_path/im2rec.py Gallery_sq.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True




# python $im2rec_path/im2rec.py baseImage_224.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
python $im2rec_path/im2rec.py challenge2.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True

# python $im2rec_path/im2rec.py lowshotImg_cropped5_224.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
# python $im2rec_path/im2rec.py lowshotImg_cropped2_224.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True
# python $im2rec_path/im2rec.py lowshotImg_cropped_224.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True

# python $im2rec_path/im2rec.py C2test.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True






# python $im2rec_path/im2rec.py MStrainFloat.lst ./ --resize=${SHAPE} --quality=${QUALITY} --color=1 --shuffle=False --center-crop=True

#PREFIX=ijba_face
#SHAPE=256
#QUALITY=90
#python $im2rec_path/im2rec.py ${PREFIX}_train.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1
#python $im2rec_path/im2rec.py ${PREFIX}_val.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1
#python $im2rec_path/im2rec.py ${PREFIX}_test.lst /home/xiong/svn/ResNet/data/IJBA/split1 --resize=${SHAPE} --quality=${QUALITY} --color=1

#python $im2rec_path/im2rec.py vgg_face_clean_val.lst ./Vgg_lmk_crop  --resize=256 --quality=90 --color=1
#python $im2rec_path/im2rec.py vgg_face_clean_test.lst ./Vgg_lmk_crop  --resize=256 --quality=90 --color=1
#$im2rec_path sf1_train.lst ./train/ sf1_train.rec resize=224 color=1 encoding='.jpg'
#$im2rec_path sf1_val.lst ./train/ sf1_val.rec resize=224 color=1 encoding='.jpg'
