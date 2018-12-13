# Training ResNet on CASIA_WEB_FACE with Softmax Loss & Center loss and Validating Models on LFW with PyTorch

This repo introduces how to train ResNet models on CASIA_WEB_FACE with Softmax Loss & Center loss and validate the models on LFW using PyTorch. With this repo as an example, you can alternatively train and test any models on any datasets for the tasks of classification and recognition.

### Pre-requisites

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* PyTorch (>=0.4)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4 NVIDIA TITAN V in parallel.

### Data Preparation

* Download the CASIA_WEB_FACE dataset for training, which contains 494,414 face images from 10,575 subjects; Download the LFW dataset for validation, which contains 13,233 face images from 5,749 subjects.
* Delete '*.DS_Store' with: `find . -name "*.DS_Store" -type f -delete`; Count class number with: `echo */ | wc`; Count image number with: `ls -lR|grep "^-"|wc -l`.
* All images (both training & validation) need to be aligned (normalized) and resized with appropriate padding. The code is in the src/Pre-_and_post-processing/FaceAlign-Resize-w-Padding.PyTorch. Training images are under 'DATA/CASIA_WEB_FACE_Aligned'. Validation images are under 'DATA/lfw_Aligned'.

### Usage

##### Training

* Configurate your training and validation settings in 'config.py'. In our case: {SEED=1337, LR_SOFTMAX=0.001, LR_CENTER=0.001, ALPHA=0.001, TRAIN_BATCH_SIZE=64, VAL_BATCH_SIZE=1, NUM_EPOCHS=90, WEIGHT_DECAY=0.0005, RGB_MEAN=\[0.485, 0.456, 0.406\], RGB_STD=\[0.229, 0.224, 0.225\], MODEL_NAME='ResNet50', TRAIN_PATH='/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE_Aligned', VAL_PATH='/home/zhaojian/zhaojian/DATA/lfw_Aligned', PAIR_TEXT_PATH='/home/zhaojian/zhaojian/DATA/pairs.txt', FILE_EXT='jpg', OPTIM='Adam'}.

* `mkdir models logs` to store the checkpoints and logs.

* Run the training script 'train.py' with `python train.py`, and your terminal will automatically display 'Epoch idx', 'Batch idx', 'Train Batch Loss', 'Train Batch Softmax Loss', 'Train Batch Center Loss', 'Train Batch Acc', 'Elapsed Time per Batch', 'Train Epoch Loss', 'Train Epoch Acc', 'Elapsed Time per Epoch','LFW VAL AUC', 'LFW VAL EER', 'Current Best val ROC AUC', 'Total Elapsed Time'.

* The checkpoints will be stored in 'models/' as '{MODEL_NAME}_CASIA-WEB-FACE-Aligned_Epoch_{epoch}_LfwAUC_{best_roc_auc}.tar', including 'epoch', 'arch', 'optim_softmax_state_dict', 'optim_center_state_dict', 'model_state_dict', 'train_epoch_loss', 'train_epoch_acc', 'best_roc_auc'.

* The training logs will be stored in 'logs/' as '{MODEL_NAME}_train_batch_loss_history_Aligned.txt' and '{MODEL_NAME}_train_batch_acc_history_Aligned.txt', including 'train_batch_loss_history' and 'train_batch_acc_history' for plotting curves.

* TODO: Train & release the ResNet-50, 101, 152 models.
