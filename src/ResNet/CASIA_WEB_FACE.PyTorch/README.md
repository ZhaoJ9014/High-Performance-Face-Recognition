# Training ResNet on CASIA_WEB_FACE and Validating Models on LFW with PyTorch

This repo shows how to train ResNet models on CASIA_WEB_FACE and validate the models on LFW using PyTorch.

### Pre-requisites

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* PyTorch (>=0.4)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4 GeForce GTX 1080Ti in parallel.

### Usage

* Dataset preparation: Download the CASIA_WEB_FACE dataset for training, which contains 494,414 face images from 10,575 subjects; Download the LFW dataset for validation, which contains 13,233 face images from 5,749 subjects.
* All images need to be aligned (normalized) and resized with appropriate padding. The code is in the src/Pre-\_and\_post-processing/FaceAlign-Resize-w-Padding.PyTorch.
