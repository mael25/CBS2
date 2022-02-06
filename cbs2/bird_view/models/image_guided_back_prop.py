import math

import numpy as np

import torch
import torch.nn as nn
################################################################
## MODIF 12/10/21
from torch.nn import functional as F
################################################################

from . import common
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle
from .ppm import PPM
from .fpn import FPN

STEPS = 5
COMMANDS = 4
DT = 0.1


class ImagePolicyModelSS(common.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, ppm_bins=None, fpn=False, all_branch=False, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)
        ################################################################
        self.fpn = None
        if fpn:
            self.fpn = FPN()
        ################################################################
        self.c = {
                'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048
                }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ################################################################
        ## MODIF 12/10/21
        self.ppm = None
        if ppm_bins is not None and len(ppm_bins) > 0:
            features_dimensions = 512 # resnet34 after layer4 has 512 features maps #original 2048
            self.ppm = PPM(features_dimensions, int(features_dimensions/len(ppm_bins)), ppm_bins)

        assert not ((self.ppm is not None) and (self.fpn is not None)), "PPM and FPN cannot be used simultaneously"
        self.has_additional_module = (self.ppm is not None) or (self.fpn is not None)
        ################################################################

        self.deconv = nn.Sequential(
            nn.BatchNorm2d((self.c if not self.has_additional_module else 2*self.c) + 128),
            nn.ConvTranspose2d((self.c if not self.has_additional_module else 2*self.c) + 128,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True),
        )

        if warp:
            ow,oh = 48,48
        else:
            ow,oh = 96,40

        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(ow,oh,STEPS),
            ) for i in range(COMMANDS)
        ])

        self.all_branch = all_branch

    def forward(self, image, velocity, command):

        if self.warp:
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = resize_images(image)
            image = torch.cat([warped_image, resized_image], 1)


        image = self.rgb_transform(image)
        ################################################################
        if self.fpn is not None:
            h = self.fpn(image)
        else:
            h = self.conv(image)

        if self.ppm is not None:
            h = self.ppm(h)
        ################################################################
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[...,None,None,None].repeat((1,128,kh,kw))

        h = torch.cat((h, velocity.cuda()), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)

        location_pred = common.select_branch(location_preds, command)

        first_pred = location_pred.data[0][0]
        diff_pred = location_pred - first_pred # offset

        mean_diff = -torch.sum(diff_pred)/5 # mean offset
        #print(mean_diff)

        return mean_diff
