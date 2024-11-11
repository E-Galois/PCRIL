# PCRIL - pytorch Implementation
# Exploiting Descriptive Completeness Prior for Cross Modal Hashing with Incomplete Labels

### 1. Introduction

This is the source code of paper "Exploiting Descriptive Completeness Prior for Cross Modal Hashing with Incomplete Labels" accepted by NeurIPS 2024.

### 2. Requirements

- torch~=2.0.1+cu118
- pillow~=9.2.0
- torchvision~=0.15.2+cu118
- tqdm~=4.64.1
- numpy~=1.24.4
- ftfy~=6.1.1
- regex~=2023.5.5
- scipy~=1.10.1
- h5py~=3.7.0
- matplotlib~=3.4.3
- ...

### 3. Preparation

#### 3.1 Download pre-trained CLIP

Pretrained CLIP model could be found in the [CLIP/clip](https://github.com/openai/CLIP/blob/main/clip). 
This code is based on the "ViT-B/32". 
You should download "ViT-B/32" and put it in the project as `./clip`

#### 3.2 Generate dataset

You should generate the following `*.mat` file for each dataset. The structure of directory `./dataset` should be:
```
    dataset
    ├── coco
    │   ├── caption.mat 
    │   ├── index.mat
    │   └── label.mat 
    ├── flickr25k
    │   ├── caption.mat
    │   ├── index.mat
    │   └── label.mat
    └── nuswide
        ├── caption.mat
        ├── index.mat 
        └── label.mat
```

Please preprocess the dataset to the appropriate input format.

More details about the generation, meaning, and format of each mat file can be found in `./dataset/README.md`.

Additionally, we will release download links for our processed clean datasets soon.

### 4. Train & Test

After preparing the Python environment, pretrained CLIP model, and dataset, we can train the PCRIL model.
#### 4.1 Train & Test on MIRFlickr25K
An example for MIRFlickr25K is provided as the current code.
``` 
python main.py
```
To run on other datasets, currently you need to change configurations in settings.py.
We will add configs for other datasets soon.
