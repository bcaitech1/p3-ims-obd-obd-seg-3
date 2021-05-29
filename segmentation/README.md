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

| Model                  |   Backbone    |
|------------------------|---------------|
| GSCNN                   |  wider_resnet101
| DeepLabV3Plus           |  EfficientNet b0
| DeepLabV3Plus           |  EfficientNet b4    
| DeepLabV3Plus           |  EfficientNet b5
| DeepLabV3Plus           |  EfficientNet b7   
| DeepLabV3Plus           |  RegNetY 320
| Unet++                  |  inceptionresnetv2
| Unet++                  |  inceptionv4
| FCN8s                   |  vgg16
| TransUNet               |  resnetV2 50
| HR+OCR Net              |  HR Net 

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

<br/><br/>

# Results

#### ✨ Best performamce of each model

| Model          | SWA | WS | GN | mAP       |
|----------------|:---:|:--:|:--:|:---------:|
| GFLv2          |     |    |    | 0.5706    |
| VFNet r2 101   | O   | O  | O  | 0.5608    |
| UniverseNet    |     |    |    | 0.5820    |

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
- Data is stored in remote server stroage.

#### Software:
- System: Ubuntu 18.04.4 LTS with Linux 4.4.0-210-generic kernel.
- Python: 3.7 distributed by Anaconda.
- CUDA: 10.1

<br/><br/>

## Reference/ Citation

[1] HR+OCRNet <br/>
```latex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
[2] [GSCNN](https://github.com/nv-tlabs/GSCNN/)<br/>
```
@article{takikawa2019gated,
  title={Gated-SCNN: Gated Shape CNNs for Semantic Segmentation},
  author={Takikawa, Towaki and Acuna, David and Jampani, Varun and Fidler, Sanja},
  journal={ICCV},
  year={2019}
}
```
[3] [TransUNet](https://github.com/Beckschen/TransUNet)<br/>
```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
<br/>
