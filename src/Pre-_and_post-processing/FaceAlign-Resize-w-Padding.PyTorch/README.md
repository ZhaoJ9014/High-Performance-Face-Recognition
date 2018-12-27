# Face Alignment and Resize with Padding with PyTorch

:triangular_flag_on_post: This repo provides a easy-to-use helper function for face alignment & resize with padding using PyTorch.

:triangular_flag_on_post: This repo provides an example on imbalanced data processing (Delete the classes with less than 10 samples).

### Pre-requisites

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* PyTorch (>=0.4)
* [MTCNN Facial Landmark Localization Tool](https://arxiv.org/pdf/1604.02878.pdf) (Install: `pip install face-alignment` or `conda install -c 1adrianb face_alignment`; Refer to this [repo](https://github.com/1adrianb/face-alignment) for more details.)

### Usage

* Organize the face images of different identities for processing under the folder 'test'. Modify the paths if needed.
* Run `face_align.py` as an example to investigate how to predict 5 faical key points, align face images and resize face images with padding.
* The aligned and resized face images will be automatically stored to the folder 'test_aligned', the subfolder names and image names remain unchanged. Modify the paths if needed.
* Run `Remove_LowShot.py` as an example to remove the low-shot classes with less than 10 samples. Modify the path and mini_num if needed.
