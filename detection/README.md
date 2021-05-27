# Detection Project for NAVER AI BoostCamp 

### 🏆Score & Standiing

(Public) 0.6051, 4등 
(Private) 0.4741, 7등

<br/><br/>

# 목차 

- [File Structure](#file-structure)
- [pipeline](#pipeline)
  - [Augmentation](#augmentation)
  - [Modeling](#Modeling)
  - [Loss](#loss)
  - [SWA](#SWA)
  - [Multiscale](#multiscale)
  - [Ensemble](#ensemble)
- [Results](#results)
- [Environment](#environment)
  - [Hardware](#hardware)
  - [Software](#software)
- [Simple Use](#simple-use)
  - [Requirements](#requirements)
  - [Install packages](#install-packages)
  - [Train](#train)
  - [Inference](#inference)

- [Reference Citation](#reference-citation)

<br/><br/>

## File Structure  

### Input
  
```
/
│ 
├── MainModel
│   ├── config
│   ├── mmdet
│   ├── requirements
│   ├── faster_rcnn_train.ipynb
│   ├── faster_rcnn_train_kfold.ipynb
│   ├── faster_rcnn_train_mosaic.ipynb
│   ├── faster_rcnn_train_ensemble.ipynb
│   ├── faster_rcnn_train_psuedo.ipynb
│   ├── gflv2_train.ipynb
│   ├── universenet_train.ipynb
│   ├── vfnet_train.ipynb
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

### Data
데이터는 상위 폴더 [README](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/README.md)에 정리되어 있음.

#### Augmentation
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
#### Train
```
bash scripts/train_detectors.sh
bash scripts/train_universenet.sh
```
#### Loss
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
#### SWA
- generalization에 강하여 test 셋에서 훨씬 좋은 성능을 보인다.
- Faster-Rcnn LB 기준 0.02AP 증가
- SWA를 mmdetection에 적용하기 쉽게 만들어진 opensource를 참고 : [Link](https://github.com/hyz-xmaster/swa_object_detection)

#### Multiscale
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
#### Ensemble
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
#### Submission preparing
```
```

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

## Model

| Method                 | mAP       |  config  |  pretrained 
|------------------------|:---------:|:--------:|:---------:
| Faster RCNN            |  0.44    |  config   |
| augmented + GFL v2 + multi scale train                |  0.49    |  config   |  pretrained 
| vfnet r2 101 + multi scale train                 |  0.5336    |  config   |  pretrained 
| vfnet r2 101 + multi scale train + SWA           |   0.5453    |  config   |  pretrained 
| vfnet r2 101 + multi scale train + SWA + WS + GN            |  0.5445    |  config   |  pretrained 
| augmented + UniverseNet + multi scale train            |  0.5820    |  config   |  pretrained 
| DetectoRS           |  0.4848    |  config   |  pretrained 
| Swin-t(30 epoch)            |  0.54    |  config   |  pretrained 
| DETR            |  0.43    |  config   |  pretrained 
| Emprical Attention            |  0.4805    |  config   |  pretrained 


## NMS(non-maximum suppression)

| nms_score_thr                 | iou_threshold     | F-mAP 
|------------------------|:---------:|:---------:
| 0.00    |  0.40    | 0.4481(채택)    
| 0.04    |  0.50    | 0.4373  
| 0.06    |  0.50    | 0.4351    
| 0.00    |  0.40    | 0.4481    
| 0.00    |  0.35    | 0.4462

## MultiScale

| Model                 | Scale     
|------------------------|:---------:
| GFLv2             |   [1333,960],[1333,800],[1333,480]  
| UniverseNet             |   [1333,960],[1333,800],[1333,480]  
| VFNet            |   [1333,800],[1333,900],[1333,1000]


<br/><br/>

## Reference/ Citation

[1] RetinaFace implementation: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)<br/>
[2] WS-DAN implementation: [GuYuc/WS-DAN.PyTorch](https://github.com/GuYuc/WS-DAN.PyTorch).<br/>
[3] EfficientNet implementation: [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).<br/>
[4] Face alignment code is from: [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/master/common/face_align.py).<br/>
