# Detection Project for NAVER AI BoostCamp 

<br/><br/>

# VumbleBot - BaselineCode  <!-- omit in toc -->

- [File Structure](#file-structure)
  - [Config](#config)
  - [mmdet](#mmdet)
  - [train jupyter baseline code](#baseline_code)
- [pipeline](#pipeline)
  - [Augmentation](#augmentation)
  - [Modeling](#Modeling)
  - [Loss](#loss)
  - [SWA](#SWA)
  - [Multiscale](#multiscale)
  - [Ensemble](#ensemble)
- [Results](#results)
- [Environment]
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
input/
│ 
├── config/ - strategies
│   ├── ST01.json
│   └── ...
│
├── checkpoint/ - checkpoints&predictions (strategy_alias_seed)
│   ├── ST01_base_00
│   │   ├── checkpoint-500
│   │   └── ...
│   ├── ST01_base_95
│   └── ...
│ 
├── data/ - competition data
│   ├── dummy_data/
│   ├── train_data/
│   └── test_data/
│
├─── embed/ - embedding caches of wikidocs.json
```

<br/><br/>

# pipeline

Augmentation
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
Train
```
bash scripts/train_detectors.sh
bash scripts/train_universenet.sh
```
Loss
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
SWA
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
Multiscale
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
Ensemble
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```
Submission preparing

<br/><br/>

## Environment

We trained models on our lab's Linux cluster. The environment listed below reflects a typical software / hardware configuration in this cluster.

Hardware:
- CPU: Xeon Gold 5120
- GPU: 2080Ti or 1080Ti
- Mem: > 64GB
- Data is stored in SSD.

Software:
- System: Ubuntu 16.04.6 with Linux 4.4.0 kernel.
- Python: 3.6 or 3.7 distributed by Anaconda.
- CUDA: 10.0

<br/><br/>

## Results

| Method                 | AP     | F-Score | x error near (m) | x error far (m) | z error near (m) | z error far (m) |
|------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 3D-LaneNet             |   89.3    | 86.4      | 0.068     | 0.477     | 0.015     | 0.202
| Gen-LaneNet            |   90.1    | 88.1      | 0.061     | 0.496     | 0.012     | 0.214

- **Rare Subset**

| Method                 | AP     | F-Score | x error near (m) | x error far (m) | z error near (m) | z error far (m) |
|------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 3D-LaneNet             |  74.6     | 72.0      | 0.166     | 0.855     | 0.039     | 0.521
| Gen-LaneNet            |  79.0     | 78.0      | 0.139     | 0.903     | 0.030     | 0.539

- **Illumination Change**

| Method                 | AP     | F-Score | x error near (m) | x error far (m) | z error near (m) | z error far (m) |
|------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 3D-LaneNet             |   74.9    | 72.5      | 0.115     | 0.601     | 0.032     | 0.230
| Gen-LaneNet            |   87.2    | 85.3      | 0.074     | 0.538     | 0.015     | 0.232

<br/><br/>

## Reference/ Citation

[1] RetinaFace implementation: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
[2] WS-DAN implementation: [GuYuc/WS-DAN.PyTorch](https://github.com/GuYuc/WS-DAN.PyTorch).
[3] EfficientNet implementation: [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).
[4] Face alignment code is from: [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/master/common/face_align.py).
