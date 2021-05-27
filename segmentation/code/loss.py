import torch.nn as nn
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt as edt
from torch.autograd import Variable
import segmentation_models_pytorch as smp
import lib.lovasz_losses as LOVASZ
from utils import get_classes_count

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
        weights = weights.to(device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, target):
        return self.CrossEntropyLoss(inputs, target)


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
        target_one_hot = make_one_hot(target)  # N, H, W -> N, C, H, W
        return self.DiceLoss(inputs_one_hot, target_one_hot)


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DiceCrossEntropyLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, target):  # N, C, H, W # N, H, W
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
        target_one_hot = make_one_hot(target)  # N, H, W -> N, C, H, W
        numerator = 2 * torch.sum(inputs_one_hot * target_one_hot)
        denominator = torch.sum(inputs_one_hot + target_one_hot)
        dice_loss = 1 - (numerator + 1) / (denominator + 1)

        return ce_loss * 1 + dice_loss * 10


class RovaszLoss(nn.Module):
    def __init__(self):
        super(RovaszLoss, self).__init__()
        self.Rovasz = LOVASZ

    def forward(self, inputs, target):
        return self.Rovasz.lovasz_softmax(inputs, target)


class RovaszCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(RovaszCrossEntropyLoss, self).__init__()
        self.Rovasz = LOVASZ
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        lovasz_loss = self.Rovasz.lovasz_softmax(inputs, target)
        ce_loss = self.CrossEntropyLoss(inputs, target)
        return lovasz_loss + ce_loss


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'weighted_cross_entropy': WeightedCrossEntropy,
    'HausdorffDT': HausdorffDTLoss,
    'dice': DiceLoss,
    'dice_cross_entropy': DiceCrossEntropyLoss,
    'rovasz': RovaszLoss,
    'rovasz_cross_entropy': RovaszCrossEntropyLoss
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
