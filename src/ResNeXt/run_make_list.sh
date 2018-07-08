#run make_list_for_extract_feature.py

NUM_SPLIT=10
# train
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 0 --data-name train
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 1 --data-name train

# test
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 0 --data-name test
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 1 --data-name test

# Gallery
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 0 --data-name Gallery
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 1 --data-name Gallery

# Probe
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 0 --data-name Probe
python make_list_for_extract_feature.py --data-dir ./data/IJBA --split-num ${NUM_SPLIT}  --out-dir ./data/IJBA --h-flip 1 --data-name Probe