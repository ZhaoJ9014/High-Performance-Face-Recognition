#!/usr/bin/env sh

#DATASHAPE=256
DATASHAPE=224
CROPSHAPE=224
# do
	# python extract_feature_resnext.py \
	# 	--data-dir extracted_feature\
	# 	--rec lowshotImg_cropped5_224.rec \
	# 	--data-type msface \
	# 	--depth 50 \
	# 	--batch-size 10 \
	# 	--model-load-epoch 87\
	# 	--load-model-prefix ./model \
	# 	--data-shape ${DATASHAPE} \
	# 	--shape ${CROPSHAPE} \
	# 	--gpus 0 \
	# 	--threads 20

	# python extract_feature_resnext.py \
	# 	--data-dir extracted_feature\
	# 	--rec baseImage_224.rec \
	# 	--data-type msface \
	# 	--depth 50 \
	# 	--batch-size 19 \
	# 	--model-load-epoch 87\
	# 	--load-model-prefix ./model \
	# 	--data-shape ${DATASHAPE} \
	# 	--shape ${CROPSHAPE} \
	# 	--gpus 0 \
	# 	--threads 20

	python extract_feature_resnext.py \
		--data-dir extracted_feature\
		--rec challenge2.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 5 \
		--model-load-epoch 87\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0 \
		--threads 20

	# python extract_feature_resnext.py \
	# 	--data-dir extracted_feature\
	# 	--rec C2test_224_test.rec \
	# 	--data-type msface \
	# 	--depth 50 \
	# 	--batch-size 4 \
	# 	--model-load-epoch 87\
	# 	--load-model-prefix ./model \
	# 	--data-shape ${DATASHAPE} \
	# 	--shape ${CROPSHAPE} \
	# 	--gpus 0 \
	# 	--threads 20























# done
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec train_224.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec train_224_h_flip.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec test.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec test_h_flip.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec Gallery.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec Gallery_h_flip.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec Probe.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
# python extract_feature_resnext.py \
#     --data-dir ./data/IJBA/split${SPLIT} \
#     --rec Probe_h_flip.rec \
#     --data-type msface \
#     --depth 50 \
#     --batch-size 256 \
#     --model-load-epoch 85\
#     --load-model-prefix ./model \
#     --data-shape ${DATASHAPE} \
#     --shape ${CROPSHAPE} \
#     --gpus 0
