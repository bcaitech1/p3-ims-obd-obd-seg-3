import numpy as np
import torch
import random
import os
from importlib import import_module
from pycocotools.coco import COCO


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """

    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_model(model_dir, num_classes, device, args, model_name, mode='train', file_name='best.pth'):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )
    if mode == 'train':
        model_path = os.path.join(model_dir, args.name, file_name)
    elif mode == 'inference':
        model_path = os.path.join(model_dir, file_name)
    print(f'loading model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# https://hoya012.github.io/blog/segmentation_tutorial_pytorch/
def copyblob(src_img, src_mask, dst_img, dst_mask, src_class, dst_class):
    mask_hist_src, _ = np.histogram(src_mask.numpy().ravel(), 11, [0, 11])
    mask_hist_dst, _ = np.histogram(dst_mask.numpy().ravel(), 11, [0, 11])
    if mask_hist_src[src_class] != 0 and mask_hist_dst[dst_class] != 0:
        """ copy src blob and paste to any dst blob"""
        mask_y, mask_x = src_mask.size()
        """ get src object's min index"""
        src_idx = np.where(src_mask == src_class)
        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]
        """ get dst object's random index"""
        dst_idx = np.where(dst_mask == dst_class)
        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx]
        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))
        for i in range(len(dst_idx[0])):
            dst_idx[0][i] = (min(dst_idx[0][i], mask_y - 1))
        for i in range(len(dst_idx[1])):
            dst_idx[1][i] = (min(dst_idx[1][i], mask_x - 1))
        dst_mask[dst_idx] = src_class
        dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]


def get_classes_count():
    # 모든 사진들로부터 background를 제외한 클래스별 픽셀 카운트를 구합니다.
    coco = COCO("../input/data/train_all.json")
    annotations = coco.loadAnns(coco.getAnnIds())
    class_num = len(coco.getCatIds())
    classes_count = [0] * class_num
    for annotation in annotations:
        class_id = annotation["category_id"]
        pixel_count = np.sum(coco.annToMask(annotation))
        classes_count[class_id] += pixel_count
    # background의 픽셀 카운트를 계산합니다.
    image_num = len(coco.getImgIds())
    total_pixel_count = image_num * 512 * 512
    background_pixel_count = total_pixel_count - sum(classes_count)
    # 모든 클래스별 픽셀 카운트를 구합니다.
    nclasses_count = [background_pixel_count] + classes_count
    return nclasses_count
