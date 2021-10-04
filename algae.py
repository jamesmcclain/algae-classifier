from typing import List

import torch
import torch.hub
import torchvision as tv


class AlgaeClassifier(torch.nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 backbone_str: str,
                 pretrained: bool,
                 prescale: int):
        super().__init__()

        self.backbone_str = backbone_str
        self.prescale = prescale

        # Backbone
        if 'efficientnet_b' in self.backbone_str:
            self.backbone = torch.hub.load(
                'lukemelas/EfficientNet-PyTorch:7e8b0d312162f335785fb5dcfa1df29a75a1783a',
                backbone_str,
                num_classes=1,
                in_channels=3,
                pretrained=('imagenet' if pretrained else None))
        else:
            backbone = getattr(tv.models, self.backbone_str)
            self.backbone = backbone(pretrained=pretrained)

        # First
        if 'efficientnet_b' in self.backbone_str:
            self.first = self.backbone._conv_stem
        else:
            if self.backbone_str == 'vgg16':
                self.first = self.backbone.features[0]
            elif self.backbone_str == 'squeezenet1_0':
                self.first = self.backbone.features[0]
            elif self.backbone_str == 'densenet161':
                self.first = self.backbone.features.conv0
            elif self.backbone_str == 'shufflenet_v2_x1_0':
                self.first = self.backbone.conv1[0]
            elif self.backbone_str == 'mobilenet_v2':
                self.first = self.backbone.features[0][0]
            elif self.backbone_str in ['mobilenet_v3_large', 'mobilenet_v3_small']:
                self.first = self.backbone.features[0][0]
            elif self.backbone_str == 'mnasnet1_0':
                self.first = self.backbone.layers[0]
            elif self.backbone_str in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                self.first = self.backbone.conv1
            else:
                raise Exception(f'Unknown backbone {self.backbone_str}')

        # Last
        if 'efficientnet_b' in self.backbone_str:
            self.last = self.backbone._fc
        elif self.backbone_str == 'vgg16':
            self.last = self.backbone.classifier[6] = torch.nn.Linear(
                in_features=4096, out_features=1, bias=True)
        elif self.backbone_str == 'squeezenet1_0':
            self.last = self.backbone.classifier[1] = torch.nn.Conv2d(
                512, 1, kernel_size=(1, 1), stride=(1, 1))
        elif self.backbone_str == 'densenet161':
            self.last = self.backbone.classifier = torch.nn.Linear(
                in_features=2208, out_features=1, bias=True)
        elif self.backbone_str == 'shufflenet_v2_x1_0':
            self.last = self.backbone.fc = torch.nn.Linear(
                in_features=1024, out_features=1, bias=True)
        elif self.backbone_str == 'mobilenet_v2':
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['mobilenet_v3_large', 'mobilenet_v3_small']:
            in_features = self.backbone.classifier[0].out_features
            self.last = self.backbone.classifier[3] = torch.nn.Linear(
                in_features=in_features, out_features=1, bias=True)
        elif self.backbone_str == 'mnasnet1_0':
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            in_features = self.backbone.fc.in_features
            self.last = self.backbone.fc = torch.nn.Linear(in_features, 1)
        else:
            raise Exception(f'Unknown backbone {self.backbone_str}')

        self.cheaplab = torch.nn.ModuleDict()
        for n in in_channels:
            self.cheaplab[str(n)] = torch.hub.load('jamesmcclain/CheapLab:master',
                                                   'make_cheaplab_model',
                                                   num_channels=n,
                                                   out_channels=3)

    def forward(self, x):
        [w, h] = x.shape[-2:]
        n = x.shape[-3]
        out = x

        if self.prescale > 1:
            out = torch.nn.functional.interpolate(
                out,
                size=[w * self.prescale, h * self.prescale],
                mode='bilinear',
                align_corners=False)
        cheaplab = self.cheaplab.get(str(n))
        if cheaplab is None:
            raise Exception(f"no CheapLab for {n} channels")
        out = cheaplab(out)
        out = self.backbone(out)

        return out


def make_algae_model(in_channels: List[int],
                     backbone_str: str,
                     pretrained: bool,
                     prescale: int):
    model = AlgaeClassifier(in_channels=in_channels,
                            backbone_str=backbone_str,
                            pretrained=pretrained,
                            prescale=prescale)
    return model
