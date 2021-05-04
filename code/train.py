import argparse
import glob
import json
import os
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import label_accuracy_score, seed_everything, seed_worker, copyblob
from loss import create_criterion

import time
from utils import load_model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    
    save_dir = increment_path(os.path.join(model_dir, args.name))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train.json / validation.json / test.json 디렉토리 설정
    train_path = data_dir + '/' + args.train
    val_path = data_dir + '/' + args.val

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # 짜다가 꼬여서 포기 albumentation용으로 Class 정의 변경해 줘야함
    # # validation 다른 aug 적용하려면 datset.py 수정 필요

    if args.aug:
        train_transform = A.Compose([
            A.CropNonEmptyMaskIfExists(200, 200, p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
            A.Resize(512, 512),
            # Normalized
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            ToTensorV2()
        ])
    val_transform = A.Compose([
        ToTensorV2()
        ])

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    transform_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    train_dataset = transform_module(data_dir=train_path, category_names=category_names, mode='train', transform=train_transform, cutmix=args.cutmix)
    val_dataset = transform_module(data_dir=val_path, category_names=category_names, mode='val', transform=val_transform, cutmix=args.cutmix)
    num_classes = train_dataset.num_classes  # 12

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            drop_last=True,
                                            worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.valid_batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            worker_init_fn=seed_worker)

    # -- model
    if args.load_model == False:
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

    else:
        if os.path.exists(os.path.join(model_dir, args.name, 'latest.pth')):
            model = load_model(model_dir, num_classes, device, args, args.model,'train', 'latest.pth').to(device)
        else:
            model = load_model(model_dir, num_classes, device, args, args.model,'train', 'best.pth').to(device)
        model = torch.nn.DataParallel(model)
        save_dir = os.path.join(model_dir, args.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'transform'), 'w') as f:
        f.write(str(train_transform) + '\n\n' + str(val_transform))

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-6
    )
    if args.scheduler == 'RP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=3,
                                                  threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                  min_lr=0, eps=1e-8, verbose=False)
    if args.scheduler == 'WSCA':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1000, T_mult=1, eta_min=0,
                                                                         last_epoch=-1, verbose=False)
    # -- logging
    start_time = time.time()
    logger = SummaryWriter(log_dir=save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    if os.path.exists(os.path.join(save_dir, 'train_info.json')):
        with open(os.path.join(save_dir, 'train_info.json'), 'r') as fp:
            train_info = json.load(fp)
    else:
        train_info = {'best_val_mIOU': 0,
                      'epoch': -1,
                      }
    start_epoch = train_info['epoch'] + 1
    best_val_mIoU = train_info['best_val_mIOU']
    print(f'best_val_mIOU: {best_val_mIoU}')
    print(f'start_epoch: {start_epoch}')
    if start_epoch >= args.epochs:
        print('already trained')
        return
    best_val_loss = np.inf
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_cnt = 0
        train_mIoU_list = []
        for idx, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            # copyblob을 만들기어 주기 위한 loop
            if args.copyblob:
                for i in range(args.batch_size):
                    rand_idx = np.random.randint(args.batch_size)
                    # category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
                    # random(?) --> background(0)
                    copyblob(src_img=images[i], src_mask=masks[i], dst_img=images[rand_idx], dst_mask=masks[rand_idx], src_class=np.random.choice([2,4,5,6,7,8], 1).item(), dst_class=0)
                    # random(?) --> paper(3)
                    copyblob(src_img=images[i], src_mask=masks[i], dst_img=images[rand_idx], dst_mask=masks[rand_idx], src_class=np.random.choice([2,4,5,6,7,8], 1).item(), dst_class=3)

            # images = torch.stack(images)        # (batch, channel, height, width)
            # masks = torch.stack(masks).long()   # (batch, channel, height, width)
            images, masks = images.to(device), masks.to(device)

            if args.model == "GSCNN":
                outputs = model(images)
                loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler != None:
                scheduler.step()
            train_loss += loss
            train_cnt += 1

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            train_mIoU_list.append(mIoU)

            if (idx + 1) % args.log_interval == 0:
                train_loss = train_loss / train_cnt
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training mIoU {np.mean(train_mIoU_list):4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/mIoU", np.mean(train_mIoU_list), epoch * len(train_loader) + idx)

                train_loss = 0
                train_cnt = 0
                train_mIoU_list = []
                
        # scheduler.step()

        # val loop
        model.eval()
        with torch.no_grad():
            print("Calculating validation results...")
            
            total_loss = 0
            cnt = 0
            mIoU_list = []
            for (images, masks, _) in val_loader:
                images = torch.stack(images)        # (batch, channel, height, width)
                masks = torch.stack(masks).long()   # (batch, channel, height, width)

                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                
                loss = criterion(outputs, masks)

                total_loss += loss
                cnt += 1
                outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

                mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
                mIoU_list.append(mIoU)

            val_loss = total_loss / cnt
            best_val_loss = min(best_val_loss, val_loss)
            val_mIoU = np.mean(mIoU_list)
            
            if val_mIoU > best_val_mIoU:
                print(f"New best model for val mIoU : {val_mIoU:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_mIoU = val_mIoU
                train_info['best_val_mIOU'] = best_val_mIoU
            # torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] mIoU : {val_mIoU:4.2%}, loss: {val_loss:4.4} || "
                f"best mIoU : {best_val_mIoU:4.2%}, best loss: {best_val_loss:4.4}"
            )
            train_info['epoch'] = epoch
            with open(os.path.join(save_dir, 'train_info.json'), 'w') as fp:
                json.dump(train_info, fp)
            torch.save(model.module.state_dict(), f"{save_dir}/latest.pth")
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/mIoU", val_mIoU, epoch)
            s = f'Time elapsed: {(time.time() - start_time)/60: .2f} min'
            print(s)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20 , help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')

    # Resize도 현재는 적용안되어 있음 (원본 size 크니깐 resize 필요해보임) Crop은 조심. 
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='FCN8s', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')

    ## 이거 수정해서 validation 나누는 기준 수정 필요
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')

    parser.add_argument('--criterion', type=str, default='rovasz_cross_entropy', help='criterion type (default: cross_entropy)')

    ### scheduler는 비활성화 해둠
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--scheduler', type=str, default=None, help='WRCA : CosineAnnealingWarmRestarts, RP : ReduceLROnPlateau')


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '../model'))
    parser.add_argument('--load_model', type=bool, default=False)

    # Special Augmentations
    parser.add_argument('--copyblob', type=bool, default=False, help='copyblob on')
    parser.add_argument('--cutmix', type=bool, default=True, help='cutmix on (default: False)')

    parser.add_argument('--train', type=str, default='train.json')
    parser.add_argument('--val', type=str, default='val.json')
    parser.add_argument('--aug', type=bool, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)


### 남은 pipline 수정사항 ####
### scheduler 지금 비활성화 (필요시 param확인후 사용)
### Validation set 나누는것 customize 필요
### loss 정의 필요   (예전 데이터 기준으로 되어있어서, Cross entropy 말고는 다시 점검해야함)
### datset.py에 load_category_names 함수 너무 비효율 적인 것 같음. 방법 필요 (수정, load_category_names, CustomDataset의 init)
### datset.py에 augmentation 정의 필요   (이거 dataset에 class로 정의하려 했는데 잘안됨, albumentation을 class에 잘 정의 필요)
### inference.py는 만들어야함!!

### 설치 필요사항  ####
### pip install tensorboard
### apt-get update
### apt-get install -y libsm6 libxext6 libxrender-dev
### pip install opencv-python

### Tesnorboard 쓰는법  ####
###### step1 : terminal에서 tensorboard --logdir='p3-ims-obd-obd-seg-3/model' --bind_all  입력
###### step2 : 각자 aistages 서버탭에 있는 tensorboard 주소로 접속

    data_dir = args.data_dir
    model_dir = args.model_dir

#     args.model = 'DeepLapV3PlusRegnety002Imagenet'
#     args.name = "DeepLapV3PlusRegnety002Imagenet"
    # args.criterion = "rovasz"

    train(data_dir, model_dir, args)
