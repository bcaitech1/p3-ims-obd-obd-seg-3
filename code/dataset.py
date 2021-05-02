# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms
import albumentations as A

import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import os

dataset_path = '/opt/ml/input/data'

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class BaseAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
            ])

    def __call__(self, image=None, mask=None):
        if mask == None:
            return self.transform(image=image)
        else:
            return self.transform(image=image, mask=masks)


class CustomDataset(Dataset):
    """COCO format"""
    num_classes = 12
    def __init__(self, data_dir, category_names, mode = 'train', transform = None, cutmix=False):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = category_names
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.image_num = len(self.coco.getImgIds())
        self.cutmix = cutmix
        
    def __getitem__(self, index: int):
        # # cutmix용 index2 (범위안에서 랜덤으로 한장 뽑기), 근데 random 보다 class 별로 확률을 주는 것도 좋을 듯
        # index2 = np.random.randint(0, self.image_num)

        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode == 'train'):
            # cutmix용 image2 load
            if self.cutmix:
                cutmix_p = np.random.rand()
                if cutmix_p > 0.7:
                    image_id2 = self.coco.getImgIds(imgIds=index2)
                    image_infos2 = self.coco.loadImgs(image_id2)[0]
                    images2 = cv2.imread(os.path.join(dataset_path, image_infos2['file_name']))
                    images2 = cv2.cvtColor(images2, cv2.COLOR_BGR2RGB).astype(np.float32)
                    images2 /= 255.0

            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            if self.cutmix:
                if cutmix_p > 0.7:
                    ann_ids2 = self.coco.getAnnIds(imgIds=image_infos2['id'])
                    anns2 = self.coco.loadAnns(ann_ids2)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # cutmix용 mask 제작
            if self.cutmix:
                if cutmix_p > 0.7:
                    masks2 = np.zeros((image_infos2["height"], image_infos2["width"]))
                    for i in range(len(anns2)):
                        className = get_classname(anns2[i]['category_id'], cats)
                        pixel_value = self.category_names.index(className)
                        masks2 = np.maximum(self.coco.annToMask(anns2[i])*pixel_value, masks2)
                    masks2 = masks2.astype(np.float32)

            # cutmix (random crop and mix) 이 코드는 두 이미지의 크기가 512로 같기 때문에 돌아갑니다.
            if self.cutmix:
                if cutmix_p > 0.7:
                    h_cut_size = np.random.randint(70, 300)
                    w_cut_size = np.random.randint(70, 300)
                    start_h1, start_w1 = np.random.randint(0, 516 - h_cut_size - 10), np.random.randint(0, 516 - w_cut_size - 10) # 잘라서 0으로
                    start_h2, start_w2 = np.random.randint(0, 516 - h_cut_size - 10), np.random.randint(0, 516 - w_cut_size - 10) # 자르기 위해 나머지를 0으로

                    masks[start_h1 : start_h1 + h_cut_size, start_w1 : start_w1 + w_cut_size] = 0
                    masks2[:start_h2,:]= 0
                    masks2[start_h2 + h_cut_size:,:]= 0
                    masks2[:,:start_w2]= 0
                    masks2[:,start_w2 + w_cut_size:]= 0
                    masks = masks + masks2
                    images[start_h1 : start_h1 + h_cut_size, start_w1 : start_w1 + w_cut_size, :] = 0
                    images2[:start_h2,:,:]= 0
                    images2[start_h2 + h_cut_size:,:,:]= 0
                    images2[:,:start_w2,:]= 0
                    images2[:,start_w2 + w_cut_size:,:]= 0
                    images = images + images2

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                images = transforms.Normalize(*self.mean_std)(images)

                masks = transformed["mask"]
            
            return images, masks, image_infos

        if (self.mode == 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                images = transforms.Normalize(*self.mean_std)(images)

                masks = transformed["mask"]
            
            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
                images = transforms.Normalize(*self.mean_std)(images)
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())