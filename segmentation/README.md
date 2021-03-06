# Segmentation Project for NAVER AI BoostCamp 

### πScore & Standiing

(Public) 0.6783, 8λ± 
(private) 0.6574, 9λ± 
<br/>

# λͺ©μ°¨ 

- [File Structure](#file-structure)
- [Simple Use](#simple-use)
  - [Requirements](#requirements)
  - [Install packages](#install-packages)
  - [Train](#train)
  - [Inference](#inference)
- [Pipeline](#pipeline)
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

### Train

```
python train.py --model 'ModelName'

#example
python train.py\
--model DeepLapV3PlusEfficientnetB4NoisyStudent\
--name effib4fold0\
--criterion cross_entropy\
--train train0.json\
--val val0.json
```

### Inference

```
#example
python inference.py\
--model DeepLapV3PlusEfficientnetB4NoisyStudent,DeepLapV3PlusEfficientnetB5NoisyStudent\
--model_dir effib4fold0,effib5fold0\
--weights 0.5,0.5
```

<br/><br/>

## File Structure  
  
```
/
βββ TransUNet
βββ code
β  βββ dataset.py
β  βββ EDA.ipynb
β  βββ inference.py
β  βββ loss.py
β  βββ model.py
β  βββ train.py
β  βββ utils.py
βββ discussion
βββ input
βββ kfold_dataset
βββ model
βββ submission
βββ README.md

```

<br/><br/>

# Pipeline

![pipeline](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/detection/pipeline1.png)

#### Data
λ°μ΄ν°λ μμ ν΄λ [README](https://github.com/bcaitech1/p3-ims-obd-obd-seg-3/blob/master/README.md)μ μ λ¦¬λμ΄ μμ.

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
train νμ΅μ΄ λ μ΄μ λμ§ μμλ 18epoch ~ 20epoch SWA optimizerλ₯Ό μ¬μ©νμ¬ test error μ΅μλ‘ weightκ° λλλ‘ μ μ©νμλ€.<br/>
μ€μ  SWA μ μ© κ²°κ³Ό Leader board score νλ½νμ¬ μ΅μ’ λͺ¨λΈμλ μ μ©νμ§ μμλ€.

<br/>

#### Modeling

| Model                  |   Backbone    |
|------------------------|---------------|
| GSCNN                   |  Wider_Resnet101
| DeepLabV3Plus           |  EfficientNet B0
| DeepLabV3Plus           |  EfficientNet B4    
| DeepLabV3Plus           |  EfficientNet B5
| DeepLabV3Plus           |  EfficientNet B7   
| DeepLabV3Plus           |  RegNetY 320
| Unet++                  |  InceptionResnetv2
| Unet++                  |  InceptionV4
| FCN8s                   |  Vgg16
| TransUNet               |  ResnetV2 50
| HR+OCR Net              |  HRNet 

<br/>

#### TTA(Augmentation)
- Horizontal Flip
- Random Brightness contrast


<br/>

#### TTA(MultiScale)

DeeplabV3+ Efficientnet-b5 20epoch κΈ°μ€

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
soft voting ensembel ν¨κ³Όλ₯Ό κ·Ήλν νκΈ°μν΄ softmax Temperatureλ₯Ό μ μ©νμλ€.<br/>
μ΅μ’ μ μΆ μ Leader board score νλ½(0.6783 -> 0.6765)νμ¬ μ΅μ’ λͺ¨λΈμ μ μ©νμ§ μμλ€.

<br/><br/>

# Results

#### β¨Final models

| Model | Architecture          | Backbone | Loss |
|------|----------------|-----|--------|
|Model1| DeepLabV3Plus  | Efficientnet B5    | LovaszLoss + CrossEntropyLoss |
|Model2| DeepLabV3Plus  | Efficientnet B5    | CrossEntropyLoss |
|Model3| DeepLabV3Plus  | Efficientnet B4    | CrossEntropyLoss |
|Model4| TransUnet      | ResnetV2 50        | CrossEntropyLoss |

<br/>

#### β¨ Ensemble

| Comb. | Weight | Public LB<br/>(mIoU) | Private LB<br/>(mIoU) |
|:-----------------------------------:|:----------------------:|:------:|:------:| 
|  Model1<br/>Model2<br/>Model3<br/>Model4  | 2.0<br/>2.0<br/>0.5<br/>1.0 | 0.6783 | 0.6574 |          

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
