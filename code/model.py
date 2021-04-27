import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp


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


# model_module = getattr(import_module("model"), args.model)

# ENCODER = 'efficientnet-b5'
# ENCODER_WEIGHTS = 'imagenet'
# #ACTIVATION = 'softmax2d'
# # DEVICE = 'cuda'
# model = smp.DeepLabV3Plus(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=12, 
# )
# # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# # model = model.to(device)

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
