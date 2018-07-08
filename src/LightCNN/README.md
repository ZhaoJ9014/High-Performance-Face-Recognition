# LightCNN for Face Recognition, implemented in TensorFlow 


### Datasets
- Training data
	- Download [MS-Celeb-1M (Aligned)](http://www.msceleb.org/download/aligned).
	- All data are RGB images and resize to 122x144.
	- Download MS-Celeb-1M cleaned image_list [10K](https://1drv.ms/t/s!AleP5K29t5x7ge87YS8Ue92h8JDDMw), [70K](https://1drv.ms/t/s!AleP5K29t5x7gfEu_3My1D3lgDhLlQ).


- Testing data
	- Download aligned LFW (122*144) [images](https://1drv.ms/u/s!AleP5K29t5x7ge88rngfpitnvpkZbw) and [list](https://1drv.ms/t/s!AleP5K29t5x7ge9DV6jfHo392ONwCA).


### Some possible solutions for improvement:
- Manaully alignment.
- Data augmentation, *e.g., Random crop.
- Ensemble different models.
- Metric learning methods.
