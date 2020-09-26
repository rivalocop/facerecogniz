# Celebrity Face Recognition

## Introduction

This consolse application is designed to train and detect celebrity's face in picture

## Requirements

1. Python 3.7
2. OpenCV
3. MXNet
4. Scikit-learn

## Installation

1. Clone this project

```bash
https://github.com/rivalocop/facerecogniz.git
```

2. Setup environment and install packages

```bash
cd facerecogniz
python3 -m venv .env
pip install -r requirements.txt
```

3. Create some neccessary folders

```bash
mkdir dataset # This folder for storing data for training
mkdir models # This folder for storing pretrained model for extract embedding
mkdir output # This folder for storing our recognize model
```

4. Extract face's embedding

I use MTCNN for face detection and this pretrained model ([here](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0)) for extracting embedding.

More information about pretrained model, please refer [here](https://github.com/deepinsight/insightface)

```bash
python extract_embedding.py
```

After this, a pickle file is created inside "output" folder which contains label and embeddings for each person we want to train.

5. Training recognize model
   Using SVM for training our classifer.

```bash
python train_model.py
```

After this, both face recognition model and label encoding is written to disk, inside folder "output". 6. Evaluate model

```bash
python demo.py
```

## Dataset

I use this [dataset](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset). It's up to you.
