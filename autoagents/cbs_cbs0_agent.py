import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string
from collections import deque
from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy

from autoagents.waypointer import Waypointer

################################################################################
# WoR
from utils.ls_fit import ls_circle as ls_circle_wor
from utils.ls_fit import project_point_to_circle, signed_angle
from cbs.models import RGBPointModel, Converter
################################################################################
# WoR
from cbs0.bird_view.models import common
from cbs0.bird_view.models.controller import CustomController, PIDController
from cbs0.bird_view.models.controller import ls_circle
from cbs0.bird_view.models.image import PPM, ImagePolicyModelSS
import torchvision.transforms as transforms
################################################################################

STEPS = 5
DT = 0.1

def get_entry_point():
    return 'CBS0Agent'

class CBS0Agent(AutonomousAgent):
    """
    CBS0 Image agent
    """

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')
        self.vizs = []
        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_evaluate')

        ########################################################################
        # WoR
        self.num_frames = 0
        self.num_cmds = 6
        self.dt = 1./20
        self.N = 10

        self.alpha_errors = deque()
        self.accel_errors = deque()

        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            #height=240-self.crop_top-self.crop_bottom, width=480,
            height=160, width=384,
            output_channel=self.num_plan*self.num_cmds
        ).to(self.device)
        self.rgb_model.load_state_dict(torch.load(self.rgb_model_dir_wor))
        self.rgb_model.eval()

        self.converter = Converter(offset=6.0, scale=[1.5, 1.5]).to(self.device)

        self.steer_points_wor = {0: 4, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3}
        self.steer_pids = {
            0 : {"Kp": 2.0, "Ki": 0.1, "Kd":0}, # Left
            1 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Right
            2 : {"Kp": 0.5, "Ki": 0.0, "Kd":0}, # Straight
            3 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Follow
            4 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Change Left
            5 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Change Right
        }
        self.accel_pids = {"Kp": 2.0, "Ki": 0.2, "Kd":0}

        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None
        ########################################################################
        # CBS
        self.model = ImagePolicyModelSS(
            backbone='resnet34',
            all_branch=False
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.rgb_model_dir))
        self.model.eval()

        self.transform = transforms.ToTensor()
        self.one_hot = torch.FloatTensor(torch.eye(4))
        self.debug = dict()

        #self.fixed_offset = float(camera_args['fixed_offset'])
        self.fixed_offset = 4.0

        w = float(384)
        h = float(160)
        self.img_size = np.array([w,h])

        #self.gap = gap
        self.gap = 5

        self.steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        pid = {
                "1" : {"Kp": 0.5, "Ki": 0.20, "Kd":0.0}, # Left
                "2" : {"Kp": 0.7, "Ki": 0.10, "Kd":0.0}, # Right
                "3" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Straight
                "4" : {"Kp": 1.0, "Ki": 0.50, "Kd":0.0}, # Follow
            }

        self.turn_control = CustomController(pid)

        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)

        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0

        self.last_brake = -1
        ########################################################################

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()

        del self.waypointer

        ########################################################################
        # WoR
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None

        self.alpha_errors.clear()
        self.accel_errors.clear()

        del self.rgb_model
        ########################################################################
        # CBS
        del self.model
        ########################################################################


    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        self.vizs.clear()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 160, 'fov': 120, 'id': f'RGB'},

        ]

        return sensors

    def run_step(self, input_data, timestamp):

        _, rgb = input_data.get(f'RGB')
        rgb = np.array(rgb[...,:3])

        # Crop images
        #_rgb = rgb[self.crop_top:-self.crop_bottom,:,:3]
        _rgb = rgb[:,:,:3]

        _rgb = _rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        speed = ego.get('spd')

        _cmd = cmd.value

        ########################################################################
        # CBS
        command = self.one_hot[_cmd - 1]
        ########################################################################
        # WoR
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3
        ########################################################################

        _rgb = torch.tensor(_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _speed = torch.tensor([speed]).float().to(self.device)

        with torch.no_grad():
            ####################################################################
            # WoR
            pred_locs = self.rgb_model(_rgb, _speed, pred_seg=False).view(self.num_cmds,self.num_plan,2)
            pred_locs = (pred_locs + 1) * self.rgb_model.img_size/2
            pred_loc = self.converter.cam_to_world(pred_locs[cmd_value])
            pred_loc = torch.flip(pred_loc, [-1])
            ####################################################################
            # CBS
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            model_pred = self.model(_rgb, _speed, _command)
            ####################################################################

        ########################################################################
        # CBS
        model_pred = model_pred.squeeze().detach().cpu().numpy()
        pixel_pred = model_pred
        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2
        steer, throt, brake = self.get_control_cbs(model_pred, _cmd, speed)
        ########################################################################
        # WoR
        steerw, throtw, brakew = self.get_control_wor(_numpy(pred_loc), cmd_value, float(spd))
        ########################################################################

        #self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), speed, cmd=_cmd))
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), speed, cmd=_cmd, pred=model_pred, controlw=(steerw, throtw, brakew), predw=pred_loc))

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    ############################################################################
    # WoR
    def get_control_wor(self, locs, cmd, spd):

        locs = np.concatenate([[[0, 0]], locs], 0)
        c, r = ls_circle_wor(locs)

        n = self.steer_points_wor.get(cmd, 1)
        closest = project_point_to_circle(locs[n], c, r)

        v = [0.0, 1.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = -signed_angle(v, w)

        # Compute steering
        self.alpha_errors.append(alpha)
        if len(self.alpha_errors) > self.N:
            self.alpha_errors.pop()

        if len(self.alpha_errors) >= 2:
            integral = sum(self.alpha_errors) * self.dt
            derivative = (self.alpha_errors[-1] - self.alpha_errors[-2]) / self.dt
        else:
            integral = 0.0
            derivative = 0.0

        steer = 0.0
        steer += self.steer_pids[cmd]['Kp'] * alpha
        steer += self.steer_pids[cmd]['Ki'] * integral
        steer += self.steer_pids[cmd]['Kd'] * derivative

        # Compute throttle and brake
        tgt_spd = np.linalg.norm(locs[:-1] - locs[1:], axis=1).mean()
        accel = tgt_spd - spd

        # Compute acceleration
        self.accel_errors.append(accel)
        if len(self.accel_errors) > self.N:
            self.accel_errors.pop()

        if len(self.accel_errors) >= 2:
            integral = sum(self.accel_errors) * self.dt
            derivative = (self.accel_errors[-1] - self.accel_errors[-2]) / self.dt
        else:
            integral = 0.0
            derivative = 0.0

        throt = 0.0
        throt += self.accel_pids['Kp'] * accel
        throt += self.accel_pids['Ki'] * integral
        throt += self.accel_pids['Kd'] * derivative

        if throt > 0:
            brake = 0.0
        else:
            brake = -throt
            throt = max(0, throt)

        if tgt_spd < 0.5:
            steer = 0.0
            throt = 0.0
            brake = 1.0

        return steer, throt, brake
    ############################################################################
    # CBS
    def get_control_cbs(self, model_pred, _cmd, speed):
        world_pred = self.unproject(model_pred)
        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)

        target_speed = np.clip(target_speed, 0.0, 5.0)

        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)

        acceleration = target_speed - speed

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        steer = self.turn_control.run_step(alpha, _cmd)
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.

        if target_speed <= self.engine_brake_threshold:
            steer = 0.0
            throttle = 0.0

        if target_speed <= self.brake_threshold:
            brake = 1.0

        self.debug = {
                # 'curve': curve,
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }

        steer, throt, brake = self.postprocess(steer, throttle, brake)
        return steer, throt, brake


    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        #control.manual_gear_shift = False

        return steer, throttle, brake

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2

        w, h = self.img_size

        f = w /(2 * np.tan(fov * np.pi / 360))

        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f

        world_z = world_y / yt
        world_x = world_z * xt

        world_output = np.stack([world_x, world_z],axis=-1)

        if self.fixed_offset:
            world_output[...,1] -= self.fixed_offset

        world_output = world_output.squeeze()

        return world_output
    ############################################################################
