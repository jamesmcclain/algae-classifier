from typing import List

import torch
import torch.hub
import torchvision as tv


# https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/9
def freeze_bn(m):
    for (name, child) in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
            child.eval()
        else:
            freeze_bn(child)


def unfreeze_bn(m):
    for (name, child) in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = True
            child.train()
        else:
            unfreeze_bn(child)


def freeze(m: torch.nn.Module) -> nn.Module:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m: torch.nn.Module) -> nn.Module:
    for p in m.parameters():
        p.requires_grad = True


class AlgaeClassifier(torch.nn.Module):
    def __init__(self,
                 in_channels: Optional[List[int]],
                 chip_size = 512,
                 num_outs: List[int] = [1],
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 canonical_size: Optional[List[int]] = None,
                 num_classes = 5):

        super().__init__()

        assert num_outs is None or isinstance(num_outs, list)
        assert backbone in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}
        assert isinstance(pretrained, bool)
        assert canonical_size is None or (isinstance(canonical_size, list) and len(canonical_size) == 2)

        self.chip_size = 512
        self.canonical_size = canonical_size
        if self.canonical_size is None:
            self.canonical_size = [chip_size, chip_size]
        self.num_outs = num_outs

        fpn = torch.hub.load(
            'jamesmcclain/pytorch-fpn:b3aa82014641c9b461f8a68bf1e73c882b9156ba',
            'make_fpn_resnet',
            name=backbone,
            fpn_type='fpn',
            num_classes=num_classes,
            fpn_channels=256,
            in_channels=3,
            out_size=self.canonical_size,
            pretrained=pretrained)
        self.backbone = fpn[0]
        self.rest = torch.nn.Sequential(fpn[1], fpn[2])

        self.pool_n = 16
        self.avgpool0 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(self.pool_n, self.pool_n))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(self.pool_n, self.pool_n))
        self.bb_features = self.backbone.m[-1][-1].conv1.in_channels
        self.seg_size = num_classes * self.pool_n * self.pool_n

        self.reset_fcs(num_outs)

        self.cheaplab = torch.nn.ModuleDict()
        for n in in_channels:
            self.cheaplab[str(n)] = torch.hub.load(
                'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
                'make_cheaplab_model',
                num_channels=n,
                out_channels=3)

        assert self.canonical_size is not None

    def reset_fcs(self, num_outs: List[int]) -> None:
        fcs = []
        self.num_outs = num_outs
        if num_outs is not None:
            for (i, num_out) in enumerate(num_outs):
                if i == 0:
                    in_features = self.bb_features + self.seg_size + self.seg_size
                    fcs.append(nn.Linear(in_features, num_out, bias=True))
                else:
                    fcs.append(nn.Linear(fcs[-1].out_features, num_out, bias=True))
        self.fcs = nn.ModuleList(fcs)

    def finetune_mode(self) -> None:
        freeze(self.backbone)
        freeze(self.rest)
        unfreeze(self.fcs)

    def whole_mode(self) -> None:
        unfreeze(self.backbone)
        unfreeze(self.rest)
        unfreeze(self.fcs)

    def freeze_bn(self) -> None:
        freeze_bn(self.backbone)
        freeze_bn(self.rest)

    def unfreeze_bn(self) -> None:
        unfreeze_bn(self.backbone)
        unfreeze_bn(self.rest)

    def forward(self, x):
        n = x.shape[-3]
        cheaplab = self.cheaplab[str(n)]
        if cheaplab is None:
            raise Exception(f"no CheapLab for {n} channels")
        x = cheaplab(x)

        x = F.interpolate(x, size=self.canonical_size, mode='bilinear', align_corners=True)
        x = self.backbone(x)
        seg_out = self.rest(x)
        seg_out = F.interpolate(seg_out, size=self.canonical_size, mode='bilinear', align_corners=True)
        x = self.avgpool0(x[-1])
        x = x.reshape(-1, self.bb_features)
        y = seg_out.softmax(dim=1)
        y0 = self.avgpool1(y).reshape(-1, self.seg_size)
        y1 = self.maxpool(y).reshape(-1, self.seg_size)
        x = torch.cat([x, y0, y1], dim=1)
        cls_out = {}
        for (i, fc) in enumerate(self.fcs):
            if i == 0:
                cls_out.update({str(i): fc(x)})
            else:
                cls_out.update({str(i): fc(cls_out.get(str(i-1)))})
        return {'seg_out': seg_out, 'cls_out': cls_out}


def make_algae_model(
        in_channels: Optional[List[int]],
        chip_size = 512,
        num_outs: List[int] = [1],
        backbone: str = 'resnet50',
        pretrained: bool = True,
        canonical_size: Optional[List[int]] = None,
        num_classes = 5):
    model = AlgaeClassifier(
        in_channels,
        chip_size,
        num_outs,
        backbone,
        pretrained,
        canonical_size,
        num_classes)
    return model
