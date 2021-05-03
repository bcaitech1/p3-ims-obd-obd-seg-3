import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm
from utils import load_model
import torchvision.transforms as transforms

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = CustomDataset.num_classes  # 18
    model_names = args.model.split(',')
    model_dirs = model_dir.split(',')
    models = []
    for model_name, model_dir in zip(model_names, model_dirs):
        models.append(load_model(model_dir, num_classes, device, args, model_name, 'inference').to(device))
    for model in models:
        model.eval()

    test_path = os.path.join(data_dir, 'test.json')

    test_transform = A.Compose([
        A.CropNonEmptyMaskIfExists(200, 200, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None,
                         always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Resize(512, 512),
        # Normalized
        ToTensorV2(),
    ])
    # test dataset
    category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
                      'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    transform_module = getattr(import_module("dataset"), args.dataset)
    test_dataset = transform_module(data_dir=test_path, category_names=category_names, mode='test',
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=1,
                                              collate_fn=collate_fn,
                                              pin_memory=use_cuda,
                                              drop_last=False,
                                              )
    print(test_loader.dataset)
    print(test_loader.dataset.mean_std)
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    softmax = torch.nn.Softmax(dim=1)
    print("Calculating inference results..")
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            print(f"step : {step} / {len(test_loader)}")
            if tta:
                for tta in range(args.tta):
                    for idx, img in enumerate(imgs):
                        img = np.array(img)
                        img = img.reshape(512, 512, 3)
                        transformed = test_transform(image=img)
                        img = transformed["image"]

                        img = transforms.Normalize(*test_loader.dataset.mean_std)(img)
                        imgs[idx] = img
                    # inference (512 x 512)
                    tta_outs = models[0](torch.stack(imgs).to(device))
                    tta_outs = softmax(tta_outs)
                    for i in range(1, len(models)):
                        tta_outs += softmax(models[i](torch.stack(imgs).to(device)))
                    if tta == 0:
                        outs = tta_outs
                    else:
                        outs += tta_outs
            else:
                outs = models[0](torch.stack(imgs).to(device))
                outs = softmax(outs)
                for i in range(1, len(models)):
                    outs += softmax(models[i](torch.stack(imgs).to(device)))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                # print(f'img shape: {img.shape}')
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array


def make_submission(data_dir, model_dir, output_dir, args):
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(data_dir, model_dir, output_dir, args)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
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
    parser.add_argument('--model', type=str, default='DeepLapV3PlusEfficientnetB5NoisyStudent',
                        help='model type (default: FCN8s)')

    parser.add_argument('--dataset', type=str, default='CustomDataset',
                        help='dataset augmentation type (default: CustomDataset)')

    parser.add_argument('--name', type=str, default='submission', help='submission file name (default: submission)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL',
                                                                        '/opt/ml/model/DeepLapV3PlusEfficientnetB5NoisyStudent'))
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/code/submission'))
    parser.add_argument('--tta', type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    # args.model = 'DeepLapV3PlusEfficientnetB5NoisyStudent'

    os.makedirs(output_dir, exist_ok=True)

    make_submission(data_dir, model_dir, output_dir, args)