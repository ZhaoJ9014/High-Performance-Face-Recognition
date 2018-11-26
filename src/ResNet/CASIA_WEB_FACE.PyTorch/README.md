# Training ResNet on CASIA_WEB_FACE and Validating Models on LFW with PyTorch

This repo shows how to train ResNet models on CASIA_WEB_FACE and validate the models on LFW using PyTorch.

### Pre-requisites

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* PyTorch (>=0.4)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4 GeForce GTX 1080Ti in parallel.

### Usage

* Dataset preparation: Download the CASIA_WEB_FACE dataset, which contains 494,414 face images from 10,575 subjects.
