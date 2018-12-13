configurations = {
    # ResNet from scratch with Adam
    1: dict(
        SEED=1337,
        LR_SOFTMAX=0.001,
        LR_CENTER=0.5,
        ALPHA = 0.01,
        TRAIN_BATCH_SIZE=256,
        VAL_BATCH_SIZE=1,
        NUM_EPOCHS=90,
        WEIGHT_DECAY=0.0005,
        RGB_MEAN=[0.485, 0.456, 0.406],
        RGB_STD=[0.229, 0.224, 0.225],
        MODEL_NAME='ResNet50',
        TRAIN_PATH='/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE_Aligned',
        VAL_PATH='/home/zhaojian/zhaojian/DATA/lfw_Aligned',
        PAIR_TEXT_PATH='/home/zhaojian/zhaojian/DATA/pairs.txt',
        FILE_EXT='jpg',  # observe, no '.' before jpg
        OPTIM='Adam',
    ),
}
