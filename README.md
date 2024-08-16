# Domain-interactive Contrastive Learning and Prototypeguided Self-training for Cross-domain Polyp Segmentation

> **Authors:**
> 
> *Ziru Lu*, 
> [*Yizhe Zhang*](https://yizhezhang.com/),
> [*Yi Zhou*](https://cse.seu.edu.cn/2021/0303/c23024a362239/page.htm), 
> [*Ye Wu*](https://dryewu.github.io/),
> and [*Tao Zhou*](https://taozh2017.github.io)

## 1. Preface

- This repository provides code for "_**Domain-interactive Contrastive Learning and Prototypeguided Self-training for Cross-domain Polyp Segmentation (DCLPS)**_" IEEE TMI 2024. 


## 2. Overview

### 2.1. Introduction
Accurate polyp segmentation plays a critical role from colonoscopy images in the diagnosis and treatment of colorectal cancer. While deep learning-based polyp segmentation models have made significant progress, they often suffer from performance degradation when applied to unseen target domain datasets collected from different imaging devices. To address this challenge, unsupervised domain adaptation (UDA) methods have gained attention by leveraging labeled source data and unlabeled target data to reduce the domain gap. However, existing UDA methods primarily focus on capturing class-wise representations, neglecting domain-wise representations. Additionally, uncertainty in pseudo labels could hinder the segmentation performance. To tackle these issues, we propose a novel Domain-interactive Contrastive Learning and Prototype-guided Self-training (DCL-PS) framework for cross-domain polyp segmentation. Specifically, domain-interactive contrastive learning (DCL) with a domain-mixed prototype updating strategy is proposed to discriminate class-wise feature representations across domains. Then, to enhance the feature extraction ability of the encoder, we present a contrastive learning-based cross-consistency training (CL-CCT) strategy, which is imposed on both the prototypes obtained by the outputs of the main decoder and perturbed auxiliary outputs. Furthermore, we propose a prototype-guided self-training (PS) strategy, which dynamically assigns a weight for each pixel during self-training, improving the quality of pseudo-labels and filtering out unreliable pixels. Experimental results demonstrate the superiority of DCL-PS in improving polyp segmentation performance in the target domain.
### 2.2. Framework Overview
<p align="center">
    <img src="imgs/framework.jpg"/> <br />
    <em> 
    Figure 1: Overview of the proposed DCL-PS.
    </em>
</p>

### 2.3. Qualitative Results
<p align="center">
    <img src="imgs/qualitative_results.jpg"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline


### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single Nvidia GeForce 3090.


1. Configuring your environment (Prerequisites):
   + Creating a virtual environment in terminal: `conda create -n DCLPS python=3.8`
   + Installing necessary packages: `pip install -r requirements.txt`.

2. Downloading necessary data:

   + DeepLab initialization can be downloaded through this [line](https://drive.google.com/file/d/1dk_4JJZBj4OZ1mkfJ-iLLWPIulQqvHQd/view?usp=sharing).
   
3. Training Configuration:
   + just run: `sh run.sh` to train our model.

4. Testing Configuration:


### 3.2 Evaluating your trained model:


### 3.3 Pre-computed maps: 
   + Pre-computed maps will be uploaded later.
## 4. Acknowledgments
This code is heavily based on the open-source implementations from [FDA](https://github.com/YanchaoYang/FDA) and [MPSCL](https://github.com/TFboys-lzz/MPSCL) 

## 5. Citation

Please cite our paper if you find the work useful: 
    
    @ARTICLE{10636198,
      author={Lu, Ziru and Zhang, Yizhe and Zhou, Yi and Wu, Ye and Zhou, Tao},
      journal={IEEE Transactions on Medical Imaging}, 
      title={Domain-interactive Contrastive Learning and Prototype-guided Self-training for Cross-domain Polyp Segmentation}, 
      year={2024},
      volume={},
      number={},
      pages={1-1},
      keywords={Prototypes;Training;Contrastive learning;Adaptation models;Uncertainty;Image segmentation;Semantics;Polyp segmentation;unsupervised domain adaptation;contrastive learning;self-training},
      doi={10.1109/TMI.2024.3443262}}

## 6. License

The source code is free for research and education use only. Any commercial use should get formal permission first.

