import cv2
import numpy as np

import torch
import torch.nn as nn

import time
from . import common
from .agent import Agent
from .controller import PIDController, CustomController
from .controller import ls_circle


class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=4.0,
                 device='cuda'):
        self._w = w
        self._h = h

    def __call__(self, map_locations):
        teacher_locations = map_locations
        teacher_locations = (teacher_locations + 1) / 2 * np.array([self._h,self._w])
        return teacher_locations.astype(int)

STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5
N_TRAFFIC_LIGHT_STATES = 1
CLASSES = {4, 6, 7, 10, 18}  # pedestrians, roadlines, roads, vehicles, tl


# def seg2D_to_ND_combined(seg, tl_info, walker_info, vehicle_info):
#     seg = seg[:, :, 0]  # CARLA stores segmentation values in R channel
#     mask = np.zeros((*seg.shape, N_CLASSES_COMBINED))
#
#     for i, seg_class in enumerate(KEEP_CLASSES):
#         if seg_class == 12:
#             mask[..., i][seg == seg_class] = tl_info
#         elif seg_class == 4:
#             mask[..., i][seg == seg_class] = walker_info
#         elif seg_class == 10:
#             mask[..., i][seg == seg_class] = vehicle_info
#         else:
#             mask[..., i][seg == seg_class] = 1
#
#     # TO_COMBINE = set(range(N_CLASSES)) - KEEP_CLASSES
#     # for i in TO_COMBINE:
#     #     mask[..., len(KEEP_CLASSES)] += seg == i
#
#     return mask
#
#
# def seg2D_to_ND(seg, tl_info, walker_info, vehicle_info, combine=False):
#     """Converts 2D segmentation image to ND array with N boolean masks.
#     Where N corresponds to number of segmentation classes."""
#     if combine:
#         return seg2D_to_ND_combined(seg, tl_info, walker_info, vehicle_info)
#
#     seg = seg[:, :, 0]  # CARLA stores segmentation values in R channel
#     mask = np.zeros((*seg.shape, N_CLASSES))
#
#     # Do not add traffic light state yet
#     for i in range(N_CLASSES-1):
#         mask[..., i][seg == i] = 1
#
#     # Add traffic light state, 12 is traffic sign class
#
#     ############################
#     # MODIF (11-09-2021)
#     #mask[..., 12][seg == 12] = tl_info #original
#     #print('\n\n Using modified segmentation for tl')
#     mask[..., 12] = tl_info
#     ############################
#
#     return mask


def regression_base():
    return nn.Sequential(
            nn.ConvTranspose2d(640,256,4,2,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,0),
            nn.BatchNorm2d(64),
            nn.ReLU(True))


def spatial_softmax_base():
    return nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ConvTranspose2d(640,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True))


class BirdViewPolicyModelSS(common.ResnetBase):
    def __init__(self, backbone='resnet18', input_channel=7, n_step=5,
                 all_branch=False, seg=False, **kwargs):
        super().__init__(backbone=backbone, input_channel=input_channel, bias_first=False)
        h = 24 if seg else 48
        w = 48

        self.deconv = spatial_softmax_base()
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(h,w,STEPS)) for i in range(COMMANDS)
        ])

        self.all_branch = all_branch

    def forward(self, bird_view, velocity, command):
        h = self.conv(bird_view)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[...,None,None,None].repeat((1,128,kh,kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)

        location_pred = common.select_branch(location_preds, command)

        if self.all_branch:
            return location_pred, location_preds

        return location_pred

# Only for phase 2 (not adapted fot Carla 0.9.10.1 yet)
# class BirdViewAgent(Agent):
#     def __init__(self, steer_points=None, pid=None, gap=5,
#                  segmentation=False, combine_seg=False,
#                  **kwargs):
#         super().__init__(**kwargs)
#
#
#         if steer_points is None:
#             # steer_points = {"1": 3, "2": 2, "3": 2, "4": 2} # Original
#             steer_points = {"1": 3, "2": 3, "3": 4, "4": 4} # GOOD 88% FullTown01-v1
#
#         if pid is None:
#             # Original
#             # pid = {
#             #     "1" : {"Kp": 1.0, "Ki": 0.1, "Kd":0}, # Left
#             #     "2" : {"Kp": 1.0, "Ki": 0.1, "Kd":0}, # Right
#             #     "3" : {"Kp": 0.8, "Ki": 0.1, "Kd":0}, # Straight
#             #     "4" : {"Kp": 0.8, "Ki": 0.1, "Kd":0}, # Follow
#             # }
#
#             # GOOD 88% FullTown01-v1
#             pid = {
#                 "1": {"Kp": 0.2, "Ki": 0.0, "Kd": 0.0},  # Left
#                 "2": {"Kp": 0.2, "Ki": 0.0, "Kd": 0.0},  # Right
#                 "3": {"Kp": 0.2, "Ki": 0.0, "Kd": 0.0},  # Straight
#                 "4": {"Kp": 0.2, "Ki": 0.0, "Kd": 0.0},  # Follow
#             }
#
#         self.steer_points = steer_points
#         self.turn_control = CustomController(pid)
#         # self.speed_control = PIDController(K_P=1.0, K_I=0.1, K_D=2.5) # Org
#         self.speed_control = PIDController(K_P=1.0, K_I=0.0, K_D=0.0) # GOOD 88% FullTown01-v1
#
#         self.seg = segmentation
#         self.gap = gap
#         self.combine_seg = combine_seg
#         self.img_size = np.array([192, 80])
#         self.fixed_offset = 4.0
#         # self.engine_brake_threshold = 1.0
#         self.brake_threshold = 0.85 # GOOD 88% FullTown01-v1
#
#         self.is_at_red_light = None
#         self.coordconverter = CoordConverter(w=192, h=80)
#
#     def unproject(self, output, world_y=1.4, fov=90):
#         cx, cy = self.img_size / 2
#
#         w, h = self.img_size
#
#         f = w / (2 * np.tan(fov * np.pi / 360))
#
#         xt = (output[..., 0:1] - cx) / f
#         yt = (output[..., 1:2] - cy) / f
#
#         world_z = world_y / yt
#         world_x = world_z * xt
#
#         world_output = np.stack([world_x, world_z], axis=-1)
#
#         if self.fixed_offset:
#             world_output[..., 1] -= self.fixed_offset
#
#         world_output = world_output.squeeze()
#
#         return world_output
#
#     def run_step(self, observations, teaching=False):
#         if self.seg:
#             birdview = seg2D_to_ND(observations['segmentation'],
#                                    observations['is_at_red_light'],
#                                    observations['is_at_walker'],
#                                    observations['is_at_vehicle'],
#                                    combine=self.combine_seg)
#             birdview = cv2.resize(birdview, (192, 80))
#             birdview = np.moveaxis(birdview, -1, 0)
#
#             self.debug['birdview_seg'] = birdview
#
#             birdview = np.expand_dims(birdview, axis=0)
#         else:
#             birdview = common.crop_birdview(observations['birdview'], dx=-10)
#
#         speed = np.linalg.norm(observations['velocity'])
#         command = self.one_hot[int(observations['command']) - 1]
#
#         with torch.no_grad():
#             if self.seg:
#                 _birdview = torch.from_numpy(birdview).float().to(self.device)
#             else:
#                 _birdview = self.transform(birdview).to(self.device).unsqueeze(0)
#             _speed = torch.FloatTensor([speed]).to(self.device)
#             _command = command.to(self.device).unsqueeze(0)
#
#             if self.model.all_branch:
#                 _locations, _ = self.model(_birdview, _speed, _command)
#             else:
#                 _locations = self.model(_birdview, _speed, _command)
#             _locations = _locations.squeeze().detach().cpu().numpy()
#
#         _map_locations = _locations
#
#         # Project back to world coordinate
#         model_pred = (_locations + 1) * self.img_size / 2
#         world_pred = self.unproject(model_pred, fov=120)
#
#         targets = [(0, 0)]
#         _cmd = int(observations['command'])
#
#         for i in range(STEPS):
#             pixel_dx, pixel_dy = world_pred[i]
#             angle = np.arctan2(pixel_dx, pixel_dy)
#             dist = np.linalg.norm([pixel_dx, pixel_dy])
#
#             targets.append([dist * np.cos(angle), dist * np.sin(angle)])
#
#         targets = np.array(targets)
#
#         target_speed = np.linalg.norm(targets[:-1] - targets[1:],
#                                       axis=1).mean() / (self.gap * DT)
#
#         c, r = ls_circle(targets)
#         n = self.steer_points.get(str(_cmd), 1)
#         closest = common.project_point_to_circle(targets[n], c, r)
#
#         acceleration = target_speed - speed
#
#         v = [1.0, 0.0, 0.0]
#         w = [closest[0], closest[1], 0.0]
#         alpha = common.signed_angle(v, w)
#
#         steer = self.turn_control.run_step(alpha, _cmd)
#         throttle = self.speed_control.step(acceleration)
#         brake = 0.0
#
#         if target_speed <= self.brake_threshold:
#             throttle = 0.0
#             steer = 0.0
#             brake = 1.0
#
#         self.debug['locations_birdview'] = _locations[:,::-1].astype(int)
#         self.debug['locations_net'] = _map_locations
#         self.debug['target'] = closest
#         self.debug['alpha'] = alpha
#         self.debug['target_speed'] = target_speed
#         self.debug['locations_pixel'] = _map_locations.astype(int)
#
#         control = self.postprocess(steer, throttle, brake)
#         if teaching:
#             return control, _map_locations
#         else:
#             return control
