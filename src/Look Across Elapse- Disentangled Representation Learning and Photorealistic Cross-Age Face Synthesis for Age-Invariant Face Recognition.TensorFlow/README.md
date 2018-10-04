
### Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition


- TensorFlow implementation of the algorithm in our under-review [Paper](https://arxiv.org/pdf/1809.00338.pdf).

<p align="center">
  <img src="pub/AIM_Results.png" width="500">
</p>

<p align="center">
  <img src="pub/AIM.png" width="500">
</p>


## Getting Started
Clone the repo:

```
git clone https://github.com/ZhaoJ9014/High_Performance_Face_Recognition/edit/master/src/Look%20Across%20Elapse-%20Disentangled%20Representation%20Learning%20and%20Photorealistic%20Cross-Age%20Face%20Synthesis%20for%20Age-Invariant%20Face%20Recognition.TensorFlow.git
```

Create a folder named 'init_model', and download the init model: [Google drive](https://drive.google.com/file/d/18an2kKQ186CuTOX4rrRXAlDaM3kH9pWu/view?usp=sharing)

FOR TRAINING:

save your training data in 'data' folder

Image name must be:
age_gender_imagename.jpg/png

(0 for male and 1 for female, both age and gender should in integer)

example run:
python main.py --dataset=sample_folder

FOR TESTING:

extract image list as in list.txt (example list.txt is provided)

run:
python tst.py


## Citation

    Jian Zhao, Yu Cheng, Yi Cheng, Yang Yang, Haochong Lan, Fang Zhao, Lin Xiong, Yan Xu, Jianshu Li, Sugiri Pranata, Shengmei Shen, Junliang Xing, Hengzhu Liu, Shuicheng Yan, and Jiashi Feng. "Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition." *Under review*, 2018.
