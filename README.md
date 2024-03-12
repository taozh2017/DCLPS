# Domain-interactive Contrastive Learning and Prototypeguided Self-training for Cross-domain Polyp Segmentation

> **Authors:** 


## 1. Preface

- This repository provides code for "_**Domain-interactive Contrastive Learning and Prototypeguided Self-training for Cross-domain Polyp Segmentation (DCLPS)**_". 


## 2. Overview

### 2.1. Introduction


### 2.2. Framework Overview



### 2.3. Qualitative Results



## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single Nvidia GeForce 3090y.


1. Configuring your environment (Prerequisites):
   
    Note that CFANet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n CFANet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

1. Downloading necessary data:

   
1. Training Configuration:


1. Testing Configuration:


### 3.2 Evaluating your trained model:

Matlab: One-key evaluation is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.
The complete evaluation toolbox (including data, map, eval code, and res): [new link](https://drive.google.com/file/d/1bnlz7nfJ9hhYsMLFSBr9smcI7k7p0pVy/view?usp=sharing). 

### 3.3 Pre-computed maps: 
They can be found in [download link](https://drive.google.com/file/d/1FY2FFDw-VLwmZ-JbJ-h4uAizcpgiY5vg/view?usp=drive_link).




## 4. Citation

Please cite our paper if you find the work useful: 
    

## 6. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---

