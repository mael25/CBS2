import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FPN(nn.Module):
    def __init__(self):
        print('\n---------------------WITH FPN-----------------------\n')
        super(FPN, self).__init__()
        self.fpn_extractor = resnet_fpn_backbone('resnet34', pretrained=True)
        # self.features = []
        # for bin in bins:
        #     self.features.append(nn.Sequential(
        #         nn.AdaptiveAvgPool2d(bin),
        #         nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(reduction_dim),
        #         nn.ReLU(inplace=True)
        #     ))
        # self.features = nn.ModuleList(self.features)

    def forward(self, x):
        fpn_maps = self.fpn_extractor(x)
        print(fpn_maps.size())
        return 0
        # x_size = x.size()
        # out = [x]
        # for f in self.features:
        #     out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        # return torch.cat(out, 1)
