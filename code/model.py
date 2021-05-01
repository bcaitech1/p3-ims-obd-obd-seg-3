import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp

import torch
import torch.nn.functional as F
from gscnn_utils.network import SEresnext
from gscnn_utils.network import Resnet
from gscnn_utils.network.wider_resnet import wider_resnet38_a2
from gscnn_utils.config import cfg
from gscnn_utils.network.mynn import initialize_weights, Norm2d
from torch.autograd import Variable

from gscnn_utils.my_functionals import GatedSpatialConv as gsc

import cv2
import numpy as np


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.pretrained_model = vgg16(pretrained=True)
        features, classifiers = list(self.pretrained_model.features.children()), list(
            self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes,
                                        1)  # input_size, output_size, kernel size  이거 나중에 upsample 전에 더해줌

        # Score pool4        
        self.score_pool4_fr = nn.Conv2d(512, num_classes,
                                        1)  # input_size, output_size, kernel size  이거 나중에 upsample 전에 더해줌

        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )

        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # UpScore2 using deconv      여기서 16 x 16을 kernel 4에 stride 2에 padding을 1로 하니깐 32 x 32이 됨  (여기서 패딩은 잘라주는 역할)
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)

        # UpScore2_pool4 using deconv     여기서 32 x 32을 kernel 4에 stride 2에 padding 1로 하니깐 64 x 64이 됨
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes,
                                                 num_classes,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)

        # UpScore8 using deconv          여기서 64 * 64을 kernel 16에 stride 8에 padding 4로 하니깐 512 x 512가 됨
        self.upscore8 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)

    def forward(self, x):
        pool3 = h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)

        h = self.conv(h)
        h = self.score_fr(h)

        score_pool3c = self.score_pool3_fr(pool3)
        score_pool4c = self.score_pool4_fr(pool4)

        # Up Score I
        upscore2 = self.upscore2(h)

        # Sum I
        h = upscore2 + score_pool4c  # 32 x 32

        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)

        # Sum II
        h = upscore2_pool4c + score_pool3c  # 64 x 64

        # Up Score III
        upscore8 = self.upscore8(h)  # 512 x 512

        return upscore8

# Custom Model Template
class DeepLapV3PlusEfficientnetB5(nn.Module):
    ENCODER = 'efficientnet-b5'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, num_classes):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusResnext50(nn.Module):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, num_classes):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusResnext101(nn.Module):
    ENCODER = 'se_resnext101_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, num_classes):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusEfficientnetB5Advprop(nn.Module):
    ENCODER = 'timm-efficientnet-b5'
    ENCODER_WEIGHTS = 'advprop'

    def __init__(self, num_classes):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusEfficientnetB5NoisyStudent(nn.Module):
    ENCODER = 'timm-efficientnet-b5'
    ENCODER_WEIGHTS = 'noisy-student'

    def __init__(self, num_classes):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusInceptionresnetv2(nn.Module):
    ENCODER = 'inceptionresnetv2'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, num_classes):
        super().__init__()
        # aux_params=dict(
        #     pooling='avg',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation='sigmoid',      # activation function, default is None
        #     classes=12,                 # define number of output labels
        # )
        self.model = smp.UnetPlusPlus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
            # aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusInceptionresnetv2Background(nn.Module):
    ENCODER = 'inceptionresnetv2'
    ENCODER_WEIGHTS = 'imagenet+background'

    def __init__(self, num_classes):
        super().__init__()
        # aux_params=dict(
        #     pooling='avg',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation='sigmoid',      # activation function, default is None
        #     classes=12,                 # define number of output labels
        # )
        self.model = smp.UnetPlusPlus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
            # aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusInceptionv4(nn.Module):
    ENCODER = 'inceptionv4'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, num_classes):
        super().__init__()
        # aux_params=dict(
        #     pooling='avg',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation='sigmoid',      # activation function, default is None
        #     classes=12,                 # define number of output labels
        # )
        self.model = smp.UnetPlusPlus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
            # aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)


class DeepLapV3PlusInceptionv4Background(nn.Module):
    ENCODER = 'inceptionv4'
    ENCODER_WEIGHTS = 'imagenet+background'

    def __init__(self, num_classes):
        super().__init__()
        # aux_params=dict(
        #     pooling='avg',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation='sigmoid',      # activation function, default is None
        #     classes=12,                 # define number of output labels
        # )
        self.model = smp.UnetPlusPlus(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=12,
            # aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)



class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, Variable(indices))
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x

class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, out_channels=1, kernel_size=kernel_sz, stride=stride,
                                                padding=upconv_pad,
                                                bias=False)
            ##doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
         

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class GSCNN(nn.Module):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self, num_classes):
        super().__init__()
        # super(GSCNN, self).__init__()
        self.num_classes = num_classes

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        
        wide_resnet = wide_resnet.module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn7 = nn.Conv2d(4096, 1, 1)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
         
        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)

    def forward(self, input):

        x_size = input.size() 

        # res 1
        m1 = self.mod1(input)

        # res 2
        m2 = self.mod2(self.pool2(m1))

        # res 3
        m3 = self.mod3(self.pool3(m2))

        # res 4-7
        m4 = self.mod4(m3)
        m5 = self.mod5(m4)
        m6 = self.mod6(m5)
        m7 = self.mod7(m6) 

        s3 = F.interpolate(self.dsn3(m3), x_size[2:],
                            mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(m4), x_size[2:],
                            mode='bilinear', align_corners=True)
        s7 = F.interpolate(self.dsn7(m7), x_size[2:],
                            mode='bilinear', align_corners=True)
        
        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)

        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s7)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        # aspp
        x = self.aspp(m7, acts)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = self.interpolate(dec0_up, m2.size()[2:], mode='bilinear',align_corners=True)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        # dec1 = self.final_seg(dec0)  
        # seg_out = self.interpolate(dec1, x_size[2:], mode='bilinear')          
        dec0_ = self.interpolate(dec0, x_size[2:], mode='bilinear')            
        seg_out = self.final_seg(dec0_)  

        return seg_out, edge_out