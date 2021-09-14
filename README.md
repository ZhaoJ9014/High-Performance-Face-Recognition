# High Performance Face Recognition
This repository provides several high performance models for unconstrained / large-scale / low-shot face recognition, based on which we have achieved:


* 2017 No.1 on ICCV 2017 MS-Celeb-1M Large-Scale Face Recognition [Hard Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1) / [Random Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1) / [Low-Shot Learning](https://www.msceleb.org/leaderboard/c2) Challenges. [WeChat News](http://mp.weixin.qq.com/s/-G94Mj-8972i2HtEcIZDpA), [NUS ECE News](http://ece.nus.edu.sg/drupal/?q=node/215), [NUS ECE Poster](https://zhaoj9014.github.io/pub/ECE_Poster.jpeg), [Award Certificate for Track-1](https://zhaoj9014.github.io/pub/MS-Track1.jpeg), [Award Certificate for Track-2](https://zhaoj9014.github.io/pub/MS-Track2.jpeg), [Award Ceremony](https://zhaoj9014.github.io/pub/MS-Awards.jpeg).


* 2017 No.1 on National Institute of Standards and Technology (NIST) IARPA Janus Benchmark A (IJB-A) Unconstrained Face [Verification](https://zhaoj9014.github.io/pub/IJBA_11_report.pdf) challenge and [Identification](https://zhaoj9014.github.io/pub/IJBA_1N_report.pdf) challenge. [WeChat News](https://mp.weixin.qq.com/s/s9H_OXX-CCakrTAQUFDm8g).


* SOTA performance on 


    * MS-Celeb-1M (Challenge1 Hard Set Coverage@P=0.95: 79.10%; Challenge1 Random Set Coverage@P=0.95: 87.50%; Challenge2 Development Set Coverage@P=0.99: 100.00%; Challenge2 Base Set Top 1 Accuracy: 99.74%; Challenge2 Novel Set Coverage@P=0.99: 99.01%).


    * IJB-A (1:1 Veification TAR@FAR=0.1: 99.6%±0.1%; 1:1 Veification TAR@FAR=0.01: 99.1%±0.2%; 1:1 Veification TAR@FAR=0.001: 97.9%±0.4%; 1:N Identification FNIR@FPIR=0.1: 1.3%±0.3%; 1:N Identification FNIR@FPIR=0.01: 5.4%±4.7%; 1:N Identification Rank1 Accuracy: 99.2%±0.1%; 1:N Identification Rank5 Accuracy: 99.7%±0.1%; 1:N Identification Rank10 Accuracy: 99.8%±0.1%).


    * Labeled Faces in the Wild (LFW) (Accuracy: 99.85%±0.217%).


    * Celebrities in Frontal-Profile (CFP) (Frontal-Profile Accuracy: 96.01%±0.84%; Frontal-Profile EER: 4.43%±1.04%; Frontal-Profile AUC: 99.00%±0.35%; Frontal-Frontal Accuracy: 99.64%±0.25%; Frontal-Frontal EER: 0.54%±0.37%; Frontal-Frontal AUC: 99.98%±0.03%).


    * CMU Multi-PIE (Rank1 Accuracy Setting-1 under ±90°: 76.12%; Rank1 Accuracy Setting-2 under ±90°: 86.73%).


### Download
- Please refer to "./src/" and this [link](https://drive.google.com/drive/folders/15L-macCtlFDondqbDDoZFcz1Qe9bLqH4?usp=sharing) to access our full source codes and models (continue to update).


****
### Donation 
:moneybag:

* Your donation is highly welcomed to help us further develop High-Performance-Face-Recognition to better facilitate more cutting-edge researches and applications on facial analytics and human-centric multi-media understanding. The donation QR code via Wechat is as below. Appreciate it very much:heart:
 
  <img src="https://github.com/ZhaoJ9014/High-Performance-Face-Recognition/blob/master/src/Donation.jpeg" width="200px"/>


### Citation
- Please consult and consider citing the following papers:


      @article{zhao2018look,
      title={Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition},
      author={Zhao, Jian and Cheng, Yu and Cheng, Yi and Yang, Yang and Lan, Haochong and Zhao, Fang and Xiong, Lin and Xu, Yan and Li, Jianshu and Pranata, Sugiri and others},
      journal={arXiv preprint arXiv:1809.00338},
      year={2018}
      }
      
      
      @article{zhao20183d,
      title={3D-Aided Dual-Agent GANs for Unconstrained Face Recognition},
      author={Zhao, Jian and Xiong, Lin and Li, Jianshu and Xing, Junliang and Yan, Shuicheng and Feng, Jiashi},
      journal={T-PAMI},
      year={2018}
      }
      
      
      @inproceedings{zhao2017dual,
      title={Dual-agent gans for photorealistic and identity preserving profile face synthesis},
      author={Zhao, Jian and Xiong, Lin and Jayashree, Panasonic Karlekar and Li, Jianshu and Zhao, Fang and Wang, Zhecan and Pranata,           Panasonic Sugiri and Shen, Panasonic Shengmei and Yan, Shuicheng and Feng, Jiashi},
      booktitle={NIPS},
      pages={66--76},
      year={2017}
      }
      
      
      @inproceedings{zhao2018towards,
      title={Towards Pose Invariant Face Recognition in the Wild},
      author={Zhao, Jian and Cheng, Yu and Xu, Yan and Xiong, Lin and Li, Jianshu and Zhao, Fang and Jayashree, Karlekar and Pranata,         Sugiri and Shen, Shengmei and Xing, Junliang and others},
      booktitle={CVPR},
      pages={2207--2216},
      year={2018}
      }
      
      
      @inproceedings{zhao3d,
      title={3D-Aided Deep Pose-Invariant Face Recognition},
      author={Zhao, Jian and Xiong, Lin and Cheng, Yu and Cheng, Yi and Li, Jianshu and Zhou, Li and Xu, Yan and Karlekar, Jayashree and       Pranata, Sugiri and Shen, Shengmei and others},
      booktitle={IJCAI},
      pages={1184--1190},
      year={2018}
      }


      @inproceedings{cheng2017know,
      title={Know you at one glance: A compact vector representation for low-shot learning},
      author={Cheng, Yu and Zhao, Jian and Wang, Zhecan and Xu, Yan and Jayashree, Karlekar and Shen, Shengmei and Feng, Jiashi},
      booktitle={ICCVW},
      pages={1924--1932},
      year={2017}
      }
