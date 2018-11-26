# Training ResNet on CASIA_WEB_FACE and Validating Models on LFW with PyTorch

This repo shows how to train ResNet models on CASIA_WEB_FACE and validate the models on LFW using PyTorch.

### Pre-requisites

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* PyTorch (>=0.4)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4 GeForce GTX 1080Ti in parallel.

### Data Preparation

* Download the CASIA_WEB_FACE dataset for training, which contains 494,414 face images from 10,575 subjects; Download the LFW dataset for validation, which contains 13,233 face images from 5,749 subjects.
* Delete "*.DS_Store" with: find . -name "*.DS_Store" -type f -delete; Count class number with: echo */ | wc; Count image numbber with: ls -lR|grep "^-"|wc -l.
* All images (both training & validation) need to be aligned (normalized) and resized with appropriate padding. The code is in the src/Pre-_and_post-processing/FaceAlign-Resize-w-Padding.PyTorch.

### Usage

##### Training

* The training script is 'train_resnet50_pretrained.py'.
* The training of ResNet is done in 3 stages (configs 1, 2 and 3 in 'config.py'), each of 30 epochs (57,960 iterations with batch_size 256). For the 1st stage, we started with the ImageNet-pre-trained model from 'torchvision.models'. After the 1st stage, we start from the saved best checkpoint model of the previous stage and divid the learning rate by a factor of 10.

