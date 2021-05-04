import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm
from utils import load_model


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = CustomDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device, args, 'inference').to(device)
    model.eval()

    test_path = os.path.join(data_dir, 'test.json')

    test_transform = A.Compose([
        ToTensorV2()
        ])

    # test dataset
    category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    transform_module = getattr(import_module("dataset"), args.dataset)
    test_dataset = transform_module(data_dir=test_path, category_names=category_names, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            pin_memory=use_cuda,
                                            drop_last=False,
                                            )

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    file_name_list = []
    preds_array = np.empty((0, size*size, 12), dtype=np.long)

    softmax_layer = nn.Softmax(dim=1)
    
    print("Calculating inference results..")
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            print(f"step : {step} / {len(test_loader)}")

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = softmax_layer(outs).detach().cpu().numpy()   # (B, C, H, W)
            oms = np.transpose(oms, (0, 2, 3, 1))               # (B, H, W, C)
            # oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)    # (32, 256, 256, 12)
            
            oms = oms.reshape([oms.shape[0], size*size, 12]).astype(float)   # (32, 256, 256, 12)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def make_preds_file(data_dir, model_dir, output_dir, args):
    # 이미 저장된 preds .npy 파일 있으면 불러오기
    if os.path.exists('/opt/ml/code/submission/file_names.npy') and os.path.exists(model_dir + '/pred_numpy.npy'):
        file_names = np.load('/opt/ml/code/submission/file_names.npy')
        preds = np.load(model_dir + '/pred_numpy.npy')
        print("pred numpy file load done!")

    # 저장된 파일 없으면 inference 해서 저장하기
    else:
        # sample_submisson.csv 열기
        submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)

        # test set에 대한 prediction
        file_names, preds = inference(data_dir, model_dir, output_dir, args)
        pred_save_path = model_dir + '/pred_numpy'

        np.save(pred_save_path, preds)
        np.save('/opt/ml/code/submission/file_names', file_names)
        print("pred numpy file save done!")
    return file_names, preds

def make_submission(file_names, preds, output_dir, args):
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)

    # # test set에 대한 prediction
    # file_names, preds = inference(data_dir, model_dir, output_dir, args)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    save_file_path = os.path.join(output_dir, f'{args.name}.csv')
    while os.path.isfile(save_file_path):
        if save_file_path[-5].isnumeric():
            save_file_path = save_file_path[:-5] + str(int(save_file_path[-5]) + 1) + ".csv"
        else:
            save_file_path = save_file_path[:-4] + str(0) + ".csv"
    submission.to_csv(save_file_path, index=False)

    print("save submission done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 512)')
    # parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='DeepLapV3PlusEfficientnetB5NoisyStudent', help='model type (default: FCN8s)')

    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset augmentation type (default: CustomDataset)')

    parser.add_argument('--name', type=str, default='submission', help='submission file name (default: submission)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/p3-ims-obd-obd-seg-3/model/normalize_DeepLabV3Plus_Effi_B5_NoisyStudent_3'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/code/submission'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    # 앙상블 할 모델명 순서에 맞게 넣기
    models = [
        'DeepLapV3PlusEfficientnetB5NoisyStudent',
        # 'DeepLapV3PlusEfficientnetB5NoisyStudent'
    ]

    os.makedirs(output_dir, exist_ok=True)

    # 앙상블 할 모델들 폴더 넣기
    ensemble_model_path_list = [
        '/opt/ml/p3-ims-obd-obd-seg-3/model/normalize_DeepLabV3Plus_Effi_B5_NoisyStudent_3',
        # '/opt/ml/p3-ims-obd-obd-seg-3/model/cutmix(rand)_DeepLabV3Plus_Effi_B5_NoisyStudent2'
    ]

    # 모델 예측값 더하기
    preds_sum = []
    for model, model_path in zip(models, ensemble_model_path_list):
        args.model = model
        model_dir = model_path
        file_names, preds_array = make_preds_file(data_dir, model_dir, output_dir, args)
        if len(preds_sum) == 0:
            preds_sum = preds_array
        else:
            preds_sum += preds_array
        
    preds = np.argmax(preds_sum, axis=-1)

    # submission 파일 만들기
    make_submission(file_names, preds, output_dir, args)
