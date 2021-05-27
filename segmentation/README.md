# Detection Project for NAVER AI BoostCamp 

### 🏆Score & Standiing

(Public) 0.6051, 4등 
(Private) 0.4741, 7등

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
  - [Set scale](#set-scale)
  - [NMS](#nms)
  - [Ensemble](#ensemble)
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

<br/>

#### Loss
- Challenge set : train size (512, 512), test size (512, 512)
- (512, 512) size를 통해 train, test를 진행하는 것 보다 논문에서 사용한 scale을 따라 진행하는 것이 더 좋은 결과가 나왔다.
- 
<br/>

#### Modeling

| Method                 | mAP       |
|------------------------|:---------:|
| DETR            |  0.43 
| Faster RCNN            |  0.44    |
| Emprical Attention            |  0.4805
| DetectoRS           |  0.4848    
| augmented + GFL v2 + multi scale train                |  0.49    
| vfnet r2 101 + multi scale train                 |  0.5336 
| Swin-t(30 epoch)            |  0.54    
| vfnet r2 101 + multi scale train + SWA + WS + GN            |  0.5445    
| vfnet r2 101 + multi scale train + SWA           |   0.5453    
| augmented + UniverseNet + multi scale train            |  0.5820    

<br/>

#### kfold
KFold(5 fold)

![kfold](https://static.packt-cdn.com/products/9781789617740/graphics/b04c27c5-7e3f-428a-9aa6-bb3ebcd3584c.png)

KFold는 전체데이터를 k개 단위로 나눠 각각을 Train과 Validation에 사용하는 기법으로 주어진 데이터 전체를 사용할 수 있다. 특히 이번 대회와 같이 데이터가 부족한 경우 도움이 된다. 이미지와 클래스 별 annotation이 5개의 폴드에 골고루 들어가도록 했는데, 이미지마다 들어있는 annotation의 갯수가 다를 수 있기 때문에 최대한 공평하게 데이터를 나누는 부분에 어려움이 있었다. 5 fold로 나눈 이유는 데이터가 가장 적은 Battery를 기준으로 5개의  fold에 이미지와 함께 annotation을 가장 골고루 나눌 수 있는 수치가 5라고 판단하였기 때문이다.

<br/>

#### Set scale
- Challenge set : train size (512, 512), test size (512, 512)
- (512, 512) size를 통해 train, test를 진행하는 것 보다 논문에서 사용한 scale을 따라 진행하는 것이 더 좋은 결과가 나왔다.
<br/>

| Model                  | Train Scale | Test Scale     
|------------------------|:---------:|:---------:
| GFLv2                  |    [(1333,960), (1333,480)]       |   [1333,960],[1333,800],[1333,480]  
| UniverseNet            |    [(1333,960), (1333,480)]       |   [1333,960],[1333,800],[1333,480]  
| VFNet                  |    [(1333,960), (1333,480)]       |   [1333,800],[1333,900],[1333,1000]
| Swin-s                 |      [(480, 1333), (512, 1333),<br/>(544, 1333), (576, 1333),<br/>(608, 1333), (640, 1333),<br/>(672, 1333), (704, 1333),<br/>(736, 1333), (768, 1333),<br/>(800, 1333)], |  [(1000, 600),(1333, 800),(1666, 1000)]

<br/>

#### NMS
(non-maximum suppression)

| nms_score_thr                 | iou_threshold     | F-mAP 
|------------------------|:---------:|:---------:
| 0.00    |  0.40    | 0.4481(채택)    
| 0.04    |  0.50    | 0.4373  
| 0.06    |  0.50    | 0.4351    
| 0.00    |  0.40    | 0.4481    
| 0.00    |  0.35    | 0.4462

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
