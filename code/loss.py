import segmentation_models_pytorch.utils.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
import numpy as np

import cv2 as cv
import numpy as np

import torch

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

from torch.autograd import Variable
import segmentation_models_pytorch as smp
import lib.lovasz_losses as LOVASZ

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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


class WeightedCrossEntropy(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropy, self).__init__()
        classes_count = get_classes_count()  # 클래스 별 카운트
        weights = torch.tensor(classes_count)
        # weights = torch.pow(weights, 1./10) # 10분의 1승 적용
        weights = torch.log(weights)  # 로그함수 적용
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print("* 클래스 별 픽셀 갯수")
        print(classes_count)
        print("* 최종 weight")
        print(weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = weights.to(device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, target):
        return self.CrossEntropyLoss(inputs, target)


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        multiple = torch.mul(log_likelihood, target)
        loss = torch.sum(multiple) / sample_num
        return loss


class focal_softCrossEntropy(nn.Module):
    def __init__(self, weight=None,
                 gamma=2.):
        self.gamma = gamma
        super(focal_softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_prob = -F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        sample_num, class_num = target.shape
        prob_focal = ((1 - prob) ** self.gamma) * log_prob

        multiple = torch.mul(log_prob, target)
        loss = torch.sum(multiple) / sample_num
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


def make_one_hot(labels, C=12):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels.to(device)
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().to(device)
    target = one_hot.scatter_(1, labels.data.to(device), 1)

    target = Variable(target)

    return target


"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        pred_ = torch.clone(pred)
        target_ = torch.clone(target)
        m = nn.Softmax(dim=1)
        pred_all = m(pred)
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        target_all = make_one_hot(target)

        loss_sum = 0

        for c in range(12):  # for class
            pred = pred_all[:, c, :, :]
            target = target_all[:, c, :, :]

            # assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
            # assert (
            #     pred.dim() == target.dim()
            # ), "Prediction and target need to be of same dimension"

            # pred = torch.sigmoid(pred)

            pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
            target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

            pred_error = (pred - target) ** 2
            distance = pred_dt ** self.alpha + target_dt ** self.alpha

            dt_field = pred_error.to(device) * distance.to(device)
            loss = dt_field.mean()

            if debug:
                loss_sum += (
                    loss.cpu().numpy(),
                    (
                        dt_field.cpu().numpy()[0, 0],
                        pred_error.cpu().numpy()[0, 0],
                        distance.cpu().numpy()[0, 0],
                        pred_dt.cpu().numpy()[0, 0],
                        target_dt.cpu().numpy()[0, 0],
                    ),
                )

            else:
                loss_sum += loss
        ce_loss_f = nn.CrossEntropyLoss()
        ce_loss = ce_loss_f(pred_, target_)
        return loss_sum / 12 / pred_all.shape[0] / 3 + ce_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.DiceLoss = smp.utils.losses.DiceLoss()

    def forward(self, inputs, target):
        # inputs: N, C(probs), H, W -> N, C(max_one_hot), H, W
        inputs_max_idx = torch.argmax(inputs, 1, keepdim=True).to(device)
        inputs_one_hot = torch.FloatTensor(inputs.shape).to(device)
        inputs_one_hot.zero_()
        inputs_one_hot.scatter_(1, inputs_max_idx, 1)
        # target: N, H, W -> H, C, H, W
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        target_one_hot = make_one_hot(target) # N, H, W -> N, C, H, W
        return self.DiceLoss(inputs_one_hot, target_one_hot)


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DiceCrossEntropyLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, target): # N, C, H, W # N, H, W
        # cross entropy loss
        ce_loss = self.CrossEntropyLoss(inputs, target)

        # dice loss
        # inputs: N, C(probs), H, W -> N, C(max_one_hot), H, W
        inputs_max_idx = torch.argmax(inputs, 1, keepdim=True).to(device)
        inputs_one_hot = torch.FloatTensor(inputs.shape).to(device)
        inputs_one_hot.zero_()
        inputs_one_hot.scatter_(1, inputs_max_idx, 1)
        # target: N, H, W -> H, C, H, W
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        target_one_hot = make_one_hot(target) # N, H, W -> N, C, H, W
        numerator = 2 * torch.sum(inputs_one_hot * target_one_hot)
        denominator = torch.sum(inputs_one_hot + target_one_hot)
        dice_loss = 1 - (numerator + 1) / (denominator + 1)


        return ce_loss*1 + dice_loss*10


def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        target = make_one_hot(target)
        return _iou(pred, target, self.size_average)

def IOU_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out


class RovaszLoss(nn.Module):
    def __init__(self):
        super(RovaszLoss, self).__init__()
        self.Rovasz = LOVASZ

    def forward(self, inputs, target):
        # inputs: N, C(probs), H, W -> N, C(max_one_hot), H, W
        # inputs_max_idx = torch.argmax(inputs, 1, keepdim=True).to(device)
        # inputs_one_hot = torch.FloatTensor(inputs.shape).to(device)
        # inputs_one_hot.zero_()
        # inputs_one_hot.scatter_(1, inputs_max_idx, 1)
        # target: N, H, W -> H, C, H, W
        # target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        # target_one_hot = make_one_hot(target) # N, H, W -> N, C, H, W
        return self.Rovasz.lovasz_softmax(inputs, target)


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'weighted_cross_entropy': WeightedCrossEntropy,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'soft_cross_entropy': softCrossEntropy,
    'focal_softCE': focal_softCrossEntropy,
    'focal2': FocalLoss2,
    'HausdorffDT': HausdorffDTLoss,
    'soft_cross_entropy': softCrossEntropy,
    'focal_softCE': focal_softCrossEntropy,
    'dice': DiceLoss,
    'dice_cross_entropy': DiceCrossEntropyLoss,
    'iou': IOU,
    'rovasz': RovaszLoss,
}


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints
