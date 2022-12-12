# μPEWFace

The *official* implementation for the ["**μPEWFace: Parallel Ensembling Weighted Deep Convolutional Neural Networks with Novel Adaptive Loss Functions for Face-based Authentication**"]()
<!-- which is accepted by [TASE 2023]()-->

>We firstly investigate and analyzes the effect of several effective loss functions based on softmax on DCNN with the ResNet50 architecture. 
>We then propose a parallel ensemble learning, namely **μPEWFace**, by taking advantage of recent novel face recognition methods: AdaFace, MagFace, ElasticFace. 
>μPEWFace elaborates on the weighted-based voting mechanism that utilizes non-optimal pre-trained models to show the proposed method’s massive potential in improving face-based authentication performa. 
>In addition, we propose to perform the matching phase for each μPEWFace model in parallel on both GPU and CPU. The results of our experiments achieve state-of-the-art figures, which show the proposed method’s massive
potential in improving face recognition performance. 

# What's New
<!-- which is accepted by [conf]()
#### [Mon. DDth, YYYY]
+ We make the repo public available.

-->



# Getting Started

## Installation
``` python
git clone https://github.com/ewigspace1910/PEWFace.git
cd PEWFace
pip install -r requirements.txt

```

## Prepare Datasets
- We use CASIA-Webface  for training and some available test sets including LFW, CFP-FP, AgeDB, CALFW, CPLFW for benchmark. 
All datasets is contributed from [Insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

- Download and extract into _data_ folder. Please unzip data and prepare like

```
PEWFace/data
├── casia-webface
│   └── 00000
│   └── 00001
│   └...
├── lfw
│   └── 00001.jpg
│   └...
├── cfp_fp
│   └── 00001.jpg
│   └...
├── ...
│
├── images_lists.txt
├── lfw_pair.txt
├── cfp_fp_pair.txt
└── ...
```

## Training 

- We re-implement MagFace, ElasticFace, AdaFace on 1 Tesla T4 GPU. We use 112x112 sized images and adopt only resnet50 architecture(with BN-Dropout-FC-BN header) for training. 
Because of 16G GPU Ram, we set batch size to 128 instead of 512 like others.

    ```python
    bash script/train.sh
    ```
## Evaluation
In this stage, we will conduct an ensemble from trained models by Weight-based Voting mechanism. Then, we apply parallel processing to the inference processing of the ensemble. 

1. Evaluate the effectiveness of Ensemble.

    - Test the individual trained model (optional):

    ```python
    python examples/test.py --c configs/softmax.yaml --p ./save/softmax/ckpt/checkpoint.pth
    ```

    - Test the Ensemble: 

    ```python
    bash script/test_ensemble.sh
    ```

2. Evaluate the effectiveness of parallel processing on both GPU and CPU.

    - Test performance of parallel processing:

    ```python
    bash script/test_parallel.sh
    ```

## Reported Results
- we will upload when the paper is published


# Citation
```

```
