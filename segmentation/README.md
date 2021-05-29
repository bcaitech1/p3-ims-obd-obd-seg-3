# Segmentation Project for NAVER AI BoostCamp 

### 🏆Score & Standiing

(Public) 0.6783, 8등 
(private) 0.6574, 9등 
<br/>

# 목차 

- [File Structure](#file-structure)
- [Simple Use](#simple-use)
  - [Requirements](#requirements)
  - [Install packages](#install-packages)
  - [Train](#train)
  - [Inference](#inference)
- [pipeline](#pipeline)
  - [Data](#data)
  - [Augmentation](#augmentation)
  - [Train](#train)
  - [Modeling](#modeling)
  - [KFold](#kfold)
  - [TTA](#tta)
  - [MultiScale](#multiscale)
  - [Softmax Temperature](#softmax-temperature)
- [Results](#results)
  - [Model](#model)
  - [NMS](#nms)
  - [MultiScale](#multiscale)
- [Environment](#environment)
  - [Hardware](#hardware)
  - [Software](#software)


- [Reference Citation](#reference-citation)

<br/><br/>

## Simple Use

### Install Requirements

```
cd ./MainModel/
pip install -r requirements.txt
```

✨apex should be installed for swin model
```
cd ./SwinModel/
pip install -r requirements.txt
```

### Train
Run each model's ipynb train file

### Inference
Run each model's ipynb inference file

<br/><br/>

## File Structure  
  
```
/
│ 
├── MainModel
│   ├── config
│   ├── mmdet
│   ├── requirements
│   ├── faster_rcnn_train.ipynb
│   ├── faster_rcnn_inference.ipynb
│   ├── faster_rcnn_train_kfold.ipynb
│   ├── faster_rcnn_train_mosaic.ipynb
│   ├── faster_rcnn_train_ensemble.ipynb
│   ├── faster_rcnn_train_psuedo.ipynb
│   ├── gflv2_train.ipynb
│   ├── gflv2_inference.ipynb
│   ├── universenet_train.ipynb
│   ├── universenet_inference.ipynb
│   ├── vfnet_train.ipynb
│   ├── vfnet_inference.ipynb
│   ├── detectoRS_train.ipynb
│   ├── detectoRS_inference.ipynb
│   └── requirements
│
├── SwinModel
│   ├── apex
│   ├── config
│   ├── mmdet
│   ├── swin_train.ipynb
│   └── swin_inference.ipynb
│ 
├── README.md
├── pipeline.png
├── __init__.py
```

<br/><br/>

# pipeline

![pipeline](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/detection/pipeline1.png)

#### Data
데이터는 상위 폴더 [README](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/README.md)에 정리되어 있음.

<br/>

#### Augmentation
- CropNonEmptyMaskIFExists
- GridDistortion
- Flip
- Resize
- GridDropout
- CoarseDropout
- Superpixel
- GridShuffle
- Copy Blob from (https://hoya012.github.io/blog/segmentation_tutorial_pytorch/)
- Cutmix

<br/>

#### kfold
KFold(5 fold)

![kfold](https://static.packt-cdn.com/products/9781789617740/graphics/b04c27c5-7e3f-428a-9aa6-bb3ebcd3584c.png)

KFold는 전체데이터를 k개 단위로 나눠 각각을 Train과 Validation에 사용하는 기법으로 주어진 데이터 전체를 사용할 수 있다. 특히 이번 대회와 같이 데이터가 부족한 경우 도움이 된다. 이미지와 클래스 별 annotation이 5개의 폴드에 골고루 들어가도록 했는데, 이미지마다 들어있는 annotation의 갯수가 다를 수 있기 때문에 최대한 공평하게 데이터를 나누는 부분에 어려움이 있었다. 5 fold로 나눈 이유는 데이터가 가장 적은 Battery를 기준으로 5개의  fold에 이미지와 함께 annotation을 가장 골고루 나눌 수 있는 수치가 5라고 판단하였기 때문이다.

<br/>


#### Loss

- Cross Entropy Loss
- Weighted Cross Entropy Loss
- Dice Loss + Cross Entropy Loss
- Lovasz Loss
- Lovasz Loss + Cross Entropy Loss
- ReduceLROnPlateau Scheduler

<br/>

#### SWA Scheduler
train 학습이 더 이상 되지 않았던 18epoch ~ 20epoch SWA optimizer를 사용하여 test error 최소로 weight가 되도록 적용하였다.
실제 SWA 적용 결과 Leader board score 하락하여 최종 모델에는 적용하지 않았다.
<br/>

#### Modeling

| Method                 | mAP       |
|------------------------|:---------:|
| GSCNN            |  0.43 
| EfficientNet b5             |  0.44    |
| EfficientNet b0           |  0.4805
| EfficientNet b4           |  0.4848    
| EfficientNet b7               |  0.49    
| RegNetY 320                |  0.5336 
| FCN8s            |  0.54    
| TransUNet            |  0.5445    
| HR+OCR Net           |   0.5453    
| augmented + UniverseNet + multi scale train            |  0.5820    

<br/>

#### TTA(Augmentation)
- Horizontal Flip
- Random Brightness contrast


<br/>

#### TTA(MultiScale)

DeeplabV3+ Efficientnet-b5 20epoch 기준

| Scale                |   weight      |    mAP    |
|-----------------------|:-------------:|:---------:|
|  Single Scale(512)   |    1   |  0.6121 |         
|  3 Multi Scale(256, 512, 1024)   | 0.3:0.3:0.3 | 0.6337 |
|  3 Multi Scale(256, 512, 1024)   | 0.3:0.4:0.3 | 0.6342 |
|  3 Multi Scale(256, 512, 1024)   | 0.3:0.5:0.3 | 0.6318 |
|  5 Multi Scale(128, 256, 512, 768, 1024)   | 0.2:0.2:0.2:0.2:0.2 | 0.6213 |
|  5 Multi Scale(128, 256, 512, 768, 1024)   | 0.15:0.2:0.3:0.2:0.15 | 0.6219 |


<br/>

#### Softmax Temperature
soft voting ensembel 효과를 극대화 하기위해 softmax Temperature를 적용하였다.
최종 제출 시 Leader board score 하락(0.6783 -> 0.6765)하여 최종 모델에 적용하지 않았다.

<br/>

#### Ensemble

| Method                |    model weight      |    mAP    |
|-----------------------|:-------------:|:---------:|
|  GFLv2, VFNe, UniversNnet   |    0.5:0.5:0.5   |  0.6048 |         
|  GFLv2, VFNet, UniverseNet Swin   | 0.5:0.5:0.5:0.5 | 0.5993 


<br/><br/>

## Environment

We trained models on our lab's Linux cluster. The environment listed below reflects a typical software / hardware configuration in this cluster.

#### Hardware:
- CPU: Xeon Gold 5120
- GPU: Tesla V100, P40
- Mem: > 90GB
- Data is stored in remote server stroage.

#### Software:
- System: Ubuntu 18.04.4 LTS with Linux 4.4.0-210-generic kernel.
- Python: 3.7 distributed by Anaconda.
- CUDA: 10.1

<br/><br/>

# Results

✨best performamce of each model

| Method                 | mAP       |  config  |  pretrained 
|------------------------|:---------:|:--------:|:---------:
| augmented + GFL v2 + multi scale train                |  0.5706    |  config   |  pretrained 
| vfnet r2 101 + multi scale train + SWA + WS + GN            |  0.5608    |  config   |  pretrained 
| augmented + UniverseNet + multi scale train            |  0.5820    |  config   |  pretrained 


<br/><br/>

## Reference/ Citation

[1] mmdetection <br/>
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}<br/>
[2] [GFL v2 & UniverseNet](https://github.com/shinya7y/UniverseNet)<br/>
