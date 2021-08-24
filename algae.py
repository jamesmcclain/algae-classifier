import torch
import torch.hub
import torchvision as tv


class AlgaeClassifier(torch.nn.Module):
    def __init__(self,
                 imagery,
                 use_cheaplab: bool = True,
                 backbone_str: str = None):
        super().__init__()

        self.imagery = imagery
        self.use_cheaplab = use_cheaplab
        self.backbone_str = backbone_str

        # Number of input bands
        if self.imagery == 'aviris':
            n = 224
        elif self.imagery == 'sentinel2':
            n = 12
        else:
            raise Exception(f'unknown imagery type {self.imagery}')

        # Number of bands going into the backbone
        if self.use_cheaplab:
            m = 2
            self.cheaplab = torch.hub.load('jamesmcclain/CheapLab:08d260b',
                                           'make_cheaplab_model',
                                           num_channels=n)
        else:
            m = n

        backbone = getattr(tv.models, self.backbone_str)
        self.backbone = backbone(pretrained=True)

        # Input and output
        if self.backbone_str == 'vgg16':
            self.first = self.backbone.features[0] = torch.nn.Conv2d(
                m, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.last = self.backbone.classifier[6] = torch.nn.Linear(
                in_features=4096, out_features=1, bias=True)
        elif self.backbone_str == 'squeezenet1_0':
            self.first = self.backbone.features[0] = torch.nn.Conv2d(
                m, 96, kernel_size=(7, 7), stride=(2, 2))
            self.last = self.backbone.classifier[1] = torch.nn.Conv2d(
                512, 1, kernel_size=(1, 1), stride=(1, 1))
        elif self.backbone_str == 'densenet161':
            self.first = self.backbone.features.conv0 = \
                torch.nn.Conv2d(
                    m,
                    96,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False)
            self.last = self.backbone.classifier = torch.nn.Linear(
                in_features=2208, out_features=1, bias=True)
        elif self.backbone_str == 'shufflenet_v2_x1_0':
            self.first = self.backbone.conv1[0] = torch.nn.Conv2d(
                m,
                24,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.fc = torch.nn.Linear(in_features=1024,
                                                           out_features=1,
                                                           bias=True)
        elif self.backbone_str == 'mobilenet_v2':
            self.first = self.backbone.features[0][0] = torch.nn.Conv2d(
                m,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['mobilenet_v3_large', 'mobilenet_v3_small']:
            self.first = self.backbone.features[0][0] = torch.nn.Conv2d(
                m,
                16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            in_features = self.backbone.classifier[0].out_features
            self.last = self.backbone.classifier[3] = torch.nn.Linear(
                in_features=in_features, out_features=1, bias=True)
        elif self.backbone_str == 'mnasnet1_0':
            self.first = self.backbone.layers[0] = torch.nn.Conv2d(
                m,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.first = self.backbone.conv1 = torch.nn.Conv2d(m,
                                                               64,
                                                               kernel_size=(7, 7),
                                                               stride=(2, 2),
                                                               padding=(3, 3),
                                                               bias=False)
            in_features = self.backbone.fc.in_features
            self.last = self.backbone.fc = torch.nn.Linear(in_features, 1)
        else:
            raise Exception(f'Unknown backbone {self.backbone_str}')

    def forward(self, x):
        out = x
        if self.use_cheaplab:
            out = self.cheaplab(out)
        out = self.backbone(out)
        return out


def make_algae_model(imagery: str, use_cheaplab: bool, backbone_str: str):
    model = AlgaeClassifier(imagery=imagery,
                            use_cheaplab=use_cheaplab,
                            backbone_str=backbone_str)
    return model
