import argparse
import os
from importlib import import_module
import pandas as pd
import numpy as np
import torch
from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_model

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_dir(folder_path):
    return '../model/' + folder_path

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = CustomDataset.num_classes  # 18
    model_names = args.model.split(',')
    model_dirs = list(map(get_model_dir, model_dir.split(',')))
    models = []
    for model_name, model_dir in zip(model_names, model_dirs):
        models.append(load_model(model_dir, num_classes, device, args, model_name, 'inference').to(device))
    for model in models:
        model.eval()

    test_path = os.path.join(data_dir, 'test.json')

    test_transform = A.Compose([
        ToTensorV2(),
    ])
    # test dataset
    category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
                      'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    transform_module = getattr(import_module("dataset"), args.dataset)
    test_dataset = transform_module(data_dir=test_path,
                                    category_names=category_names,
                                    mode='test',
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=1,
                                              collate_fn=collate_fn,
                                              pin_memory=use_cuda,
                                              drop_last=False,
                                              )

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    softmax = torch.nn.Softmax(dim=1)
    print("Calculating inference results..")
    weights = list(map(float, args.weights.split(',')))
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            print(f"step : {step} / {len(test_loader)}")

            images = torch.stack(imgs)  # (batch, channel, height, width)
            images = images.to(device)
            images2 = torch.nn.UpsamplingBilinear2d(size=(256, 256))(images)
            images3 = images
            images5 = torch.nn.UpsamplingBilinear2d(size=(1024, 1024))(images)
            outputs2 = models[0](images2)
            outputs2 = torch.nn.UpsamplingBilinear2d(size=(512, 512))(outputs2)
            outputs3 = models[0](images3)
            outputs5 = models[0](images5)
            outputs5 = torch.nn.UpsamplingBilinear2d(size=(512, 512))(outputs5)
            outputs = (outputs2 * 0.3) + (outputs3 * 0.4) + (outputs5 * 0.3)
            outs = softmax(outputs) * weights[0]
            for i in range(1, len(models)):
                outputs2 = models[i](images2)
                outputs2 = torch.nn.UpsamplingBilinear2d(size=(512, 512))(outputs2)
                outputs3 = models[i](images3)
                outputs5 = models[i](images5)
                outputs5 = torch.nn.UpsamplingBilinear2d(size=(512, 512))(outputs5)
                outputs = (outputs2 * 0.3) + (outputs3 * 0.4) + (outputs5 * 0.3)
                outs += softmax(outputs) * weights[i]
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
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
    print(f'file path: {save_file_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--model', type=str, default='DeepLapV3PlusEfficientnetB5NoisyStudent', help='model type (default: DeepLapV3PlusEfficientnetB5NoisyStudent)')
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset augmentation type (default: CustomDataset)')
    parser.add_argument('--name', type=str, default='submission', help='submission file name (default: submission)')

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '../input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '../model/DeepLapV3PlusEfficientnetB5NoisyStudent'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '../submission'))
    parser.add_argument('--tta', type=int, default=0)
    parser.add_argument('--weights', type=str, default='')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    make_submission(data_dir, model_dir, output_dir, args)
