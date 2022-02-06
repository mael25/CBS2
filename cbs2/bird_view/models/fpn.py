import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# FPN feature extractor relying on ResNet34 as backbone, followed by a
# down-sampling of the obtained multi-scale merged feature maps using max pool,
# so they all have the dimensions as the last ResNet convolutional layer. They are
# finally concatenated together, which results in an output of the same dimensions
# as ResNet34, but with twice as much channels (1042 instead of 512).

class FPN(nn.Module):
    def __init__(self):
        #print('\n---------------------WITH FPN-----------------------\n')
        super(FPN, self).__init__()
        self.fpn_extractor = resnet_fpn_backbone('resnet34', pretrained=True)
        self.downsamplers = []
        for i in range(4):
            self.downsamplers.append(nn.MaxPool2d((pow(2,3-i))))
        self.downsamplers = nn.ModuleList(self.downsamplers)

    def forward(self, x):
        fpn_maps = self.fpn_extractor(x)
        out = []
        for i, ds in enumerate(self.downsamplers):
            out.append(ds(fpn_maps[str(i)]))
        return torch.cat(out, 1)
