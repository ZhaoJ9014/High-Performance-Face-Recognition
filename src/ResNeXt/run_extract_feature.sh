#!/usr/bin/env sh

#DATASHAPE=256
DATASHAPE=224
CROPSHAPE=224

for i in $( seq 1 10 )
do
#SPLIT=10

# task(){
#     echo "$0";
#     '''python extract_feature_resnext.py \'''
#     python extract_feature_resnet.py \
#         --data-dir ./data/IJBA/split${SPLIT} \
#         --rec Gallery_h_flip.rec \
#         --data-type vggface \
#         --depth 50 \
#         --batch-size 256 \
#         --model-load-epoch 125\
#         --load-model-prefix ./model \
#         --data-shape ${DATASHAPE} \
#         --shape ${CROPSHAPE} \
#         --gpus $0
# }
# for index in 0; do
#     task "$index" &
# done
# wait

	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec train_224.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec train_224_h_flip.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec test.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec test_h_flip.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec Gallery.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec Gallery_h_flip.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec Probe.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0
	python extract_feature_resnet.py \
		--data-dir ./data/IJBA/split${i} \
		--rec Probe_h_flip.rec \
		--data-type msface \
		--depth 50 \
		--batch-size 256 \
		--model-load-epoch 125\
		--load-model-prefix ./model \
		--data-shape ${DATASHAPE} \
		--shape ${CROPSHAPE} \
		--gpus 0

done
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
