import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from .pytorch_grad_cam import GuidedBackpropReLUModel
from .pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
from .pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

import cbs2.bird_view.models.image_guided_back_prop as LBSimage

def get_gb(rgb, architecture, velocity, command):

    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'backbone': 'resnet34'
    }

    if architecture=='PPM':
        student_ppm = LBSimage.ImagePolicyModelSS(
                      config['backbone'],
                      ppm_bins=[1, 2, 3, 6]
                      ).to(config['device'])
        ckpt_ppm = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_ppm_l2/model-90.th'
        student_ppm.load_state_dict(torch.load(ckpt_ppm))
        model = student_ppm
    if architecture=='FPN':
        student_fpn = LBSimage.ImagePolicyModelSS(
                      config['backbone'],
                      fpn=True
                      ).to(config['device'])
        ckpt_fpn = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_fpn_l2/model-100.th'
        student_fpn.load_state_dict(torch.load(ckpt_fpn))
        model = student_fpn
    else:
        student_orig = LBSimage.ImagePolicyModelSS(
                      config['backbone']
                      ).to(config['device'])
        ckpt_orig = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_original_l2/model-86.th'
        student_orig.load_state_dict(torch.load(ckpt_orig))
        model = student_orig

    #print(model)

    rgb_img_cv2 = rgb[:, :, ::-1]
    rgb_img = np.float32(rgb_img_cv2) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    font = cv2.FONT_HERSHEY_SIMPLEX
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=None, velocity=velocity, command=command)
    gb = deprocess_image(gb)
    gb_viz = cv2.putText(gb.copy(), architecture, (10,10), font, 0.3, (255, 255, 255), 1, cv2.LINE_8)

    return gb_viz
