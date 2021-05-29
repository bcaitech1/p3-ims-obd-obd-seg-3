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
  - [SWA](#SWA)
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
└── __init__.py
```

<br/><br/>

# pipeline

![pipeline](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/detection/pipeline_image.png)

#### Data
데이터는 상위 폴더 [README](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/README.md)에 정리되어 있음.

<br/>

#### Augmentation

- Cutout
- RandomBrightnessContrast
- ImageCompression
- ChannelShuffle
- RGBShift
- HueSaturationValue
- RandomGamma
- CLAHE
- RandomRotate90
- Blur : Blur, MedianBlur, MotionBlur, GaussNoise
- Flip

<br/>

#### Modeling

| Model               | Multi Scale<br/>Train | SWA | WS | GN | mAP       |
|------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|
| DETR            |  ||||0.4300| 
| Faster RCNN         |  ||| |  0.4400    |
| Emprical Attention |      |||     |  0.4805
| DetectoRS         ||||  |  0.4848    
| GFLv2        |✔|||         |  0.4900    
| Vfnet r2 101    |✔|||              |  0.5336 
| Swin-t          |✔||||  0.5400    
| Vfnet r2 101    |✔|✔|✔|✔ |  0.5445    
| Vfnet r2 101    |✔|✔| |  |   0.5453    
| UniverseNet     |✔| | |  | 0.5820    

<br/>

#### SWA (Stochastic Weight Averaging)
- Generalization에 강하여 test 셋에서 훨씬 좋은 성능을 보인다.</br>
- SWA를 mmdetection에 적용하기 쉽게 만들어진 opensource를 참고 : [Link](https://github.com/hyz-xmaster/swa_object_detection)
- Faster-Rcnn LB 기준 0.02 증가

<br/>

#### Set scale
- Challenge set : train size (512, 512), test size (512, 512)
- (512, 512) size를 통해 train, test를 진행하는 것 보다 논문에서 사용한 scale을 따라 진행하는 것이 더 좋은 결과가 나왔다.
<br/>

| Model                  | Train Scale | Test Scale     
|------------------------|:-----------:|:-----------:
| GFLv2                  |    [(1333,960), (1333,480)]       |   [(1333,960),(1333,800),(1333,480)]
| UniverseNet            |    [(1333,960), (1333,480)]       |   [(1333,960),(1333,800),(1333,480)]
| VFNet                  |    [(1333,960), (1333,480)]       |   [(1333,800),(1333,900),(1333,1000)]
| Swin-s                 |      [(480, 1333), (512, 1333),<br/>(544, 1333), (576, 1333),<br/>(608, 1333), (640, 1333),<br/>(672, 1333), (704, 1333),<br/>(736, 1333), (768, 1333),<br/>(800, 1333)], |  [(1000, 600),(1333, 800),(1666, 1000)]

<br/>

#### NMS (Non-Maximum Suppression)

| nms_score_thr                 | iou_threshold     | F-mAP 
|:--------------:|:---------:|:---------:
| 0.00    |  0.35    | 0.4462
|  _**0.00**_    |  _**0.40**_    | _**0.4481**_
| 0.04    |  0.50    | 0.4373  
| 0.06    |  0.50    | 0.4351    
 

<br/><br/>

# Results

#### ✨ Best performamce of each model

| Model                 | SWA | WS | GN | mAP       |
|-----------------------|:---:|:--:|:--:|:---------:|
| augmented + GFLv2     |     |    |    | 0.5706    |
| VFNet r2 101          | ✔ | ✔  | ✔  | 0.5608    |
| augmented + UniverseNet|    |    |    | 0.5820    |

<br/>

#### ✨ Ensemble

| Method                            | Ensemble ratio       | mAP    |
|-----------------------------------|----------------------|:------:|
|  GFLv2, VFNet, UniverseNet        | 0.5, 0.5, 0.5        | 0.6048 |         
|  GFLv2, VFNet, UniverseNet, Swin  | 0.5, 0.5, 0.5, 0.5   | 0.5993 |

<br/><br/>

## Environment

We trained models on our lab's Linux cluster. The environment listed below reflects a typical software / hardware configuration in this cluster.

#### Hardware:
- CPU: Xeon Gold 5120
- GPU: Tesla V100, P40
- Mem: > 90GB
- Data is stored in remote server storage.

#### Software:
- System: Ubuntu 18.04.4 LTS with Linux 4.4.0-210-generic kernel.
- Python: 3.7 distributed by Anaconda.
- CUDA: 10.1

<br/><br/>

## Reference/ Citation

[1] mmdetection <br/>
```
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
}
```
<br/>
[2] [GFL v2 & UniverseNet](https://github.com/shinya7y/UniverseNet)<br/>
[3] [VFNET](https://github.com/hyz-xmaster/VarifocalNet)<br/>
[4] [DetectoRS](https://github.com/joe-siyuan-qiao/DetectoRS)<br/>
[5] [SWIN](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)<br/>
