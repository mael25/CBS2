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

from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs

from rails.models import EgoModel, CameraModel
from autoagents.waypointer import Waypointer

from cbs2.bird_view.models import common
from cbs2.bird_view.models.controller import CustomController, PIDController
from cbs2.bird_view.models.controller import ls_circle
from cbs2.bird_view.models.image import PPM, ImagePolicyModelSS
import torchvision.transforms as transforms

STEPS = 5
DT = 0.1

def get_entry_point():
    return 'ImageAgent'

class ImageAgent(AutonomousAgent):

    """
    Trained image agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        if hasattr(self, 'ppm_bins'):
            self.ppm_bins = list(map(int, self.ppm_bins.split("-"))) # "1-2-3-6" --> [1, 2, 3, 6]
        else:
            self.ppm_bins = None

        if not hasattr(self, 'fpn'):
            self.fpn = False

        self.device = torch.device('cuda')

        self.image_model = CameraModel(config).to(self.device)
        self.image_model.load_state_dict(torch.load(self.main_model_dir))
        self.image_model.eval()

        self.vizs = []

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project= path_to_conf_file.split('/')[-1].split('.')[0])

################################################################################
# CBS
        self.model = ImagePolicyModelSS(
            backbone='resnet34',
            all_branch=False,
            ppm_bins=self.ppm_bins,
            fpn=self.fpn
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

        self.engine_brake_threshold_straight = 3.8
        self.brake_threshold_straight = 3

        #self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0

        self.last_brake = -1
################################################################################

        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None

        del self.waypointer
        del self.image_model

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
            {'type': 'sensor.stitch_camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB'},
            #CBS
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 160, 'fov': 120, 'id': f'RGB'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):

        _, wide_rgb = input_data.get(f'Wide_RGB')
        _, narr_rgb = input_data.get(f'Narrow_RGB')

        # Crop images
        _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
        _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]

        _wide_rgb = _wide_rgb[...,::-1].copy()
        _narr_rgb = _narr_rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        ########################################################################
        #CBS
        _, rgb = input_data.get(f'RGB')
        rgb = np.array(rgb[...,:3])
        _rgb = rgb[:,:,:3]
        _rgb = _rgb[...,::-1].copy()
        ########################################################################

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        spd = ego.get('spd')

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

        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)

        if self.all_speeds:
            steer_logits, throt_logits, brake_logits = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value)
            # Interpolate logits
            steer_logit = self._lerp(steer_logits, spd)
            throt_logit = self._lerp(throt_logits, spd)
            brake_logit = self._lerp(brake_logits, spd)
        else:
            steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))


        action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

        brake_prob = float(action_prob[-1])

        steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
        throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

        steer_wor, throt_wor, brake_wor = self.post_process(steer, throt, brake_prob, spd, cmd_value)


        rgb_wor = np.concatenate([wide_rgb, narr_rgb[...,:3]], axis=1)

        ########################################################################
        #CBS
        speed = ego.get('spd')

        # Test 29 dec - outdated, replaced by offset given only to the network
        #speed = speed +1.6
        # if timestamp <3:
        #     speed=3.0

        # Issue when zero speed fed to network: waypoints lead to a stop.
        # Thus, feed slightly higher speed to network so that it is just able to move
        # If there is an obstacle, this offset is not enough to make it move


        # if speed < 2:
        #     adapted_speed = speed + 1.6
        # else:
        #     adapted_speed = speed
        adapted_speed = 0.0

        _cmd = cmd.value
        command = self.one_hot[_cmd - 1]

        _rgb = torch.tensor(_rgb[None]).float().permute(0,3,1,2).to(self.device)
        #print(f'RGB size: {_rgb.size()}')


        #_speed = torch.tensor([speed]).float().to(self.device) #original
        #_speed = torch.tensor([speed+1.6]).float().to(self.device) #29dec
        _speed = torch.tensor([adapted_speed]).float().to(self.device)

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            #_speed = torch.FloatTensor([speed]).to(self.device) #original
            #_speed = torch.FloatTensor([speed+1.6]).to(self.device)
            _speed = torch.FloatTensor([adapted_speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            model_pred = self.model(_rgb, _speed, _command)

        model_pred = model_pred.squeeze().detach().cpu().numpy()
        pixel_pred = model_pred
        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2
        steer, throt, brake, target_speed = self.get_control(model_pred, _cmd, speed)

        #self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), speed, cmd=_cmd))
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), speed, target_speed=target_speed, cmd=_cmd, pred=model_pred))
        ########################################################################

        #self.vizs.append(visualize_obs(rgb_wor, 0, (steer, throt, brake), spd, cmd=cmd_value+1))

        if len(self.vizs) > 3000:
            self.flush_data()

        self.num_frames += 1

        # We send the Wor controls but plot CBS2 predictions and infos
        return carla.VehicleControl(steer=steer_wor, throttle=np.max([throt_wor, 0.2]), brake=0.0)

    def _lerp(self, v, x):
        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):

        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1

        if spd > {0:10,1:10}.get(cmd, 20)/3.6: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing

        return steer, throt, brake
################################################################################
# CBS
    def get_control(self, model_pred, _cmd, speed):
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

        #steer = self.turn_control.run_step(alpha, _cmd) #original - outdated since new dataset
        steer = self.turn_control.run_step(alpha, _cmd)/3 #29dec
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.

        # if target_speed <= self.engine_brake_threshold:
        #     steer = 0.0
        #     throttle = 0.0
        #
        # if target_speed <= self.brake_threshold:
        #     brake = 1.0

        # As we go faster when we go straight, we have different stopping threshold
        if np.abs(steer)<0.05:
            if target_speed <= self.engine_brake_threshold_straight:
                throttle = 0.0
            if target_speed <= self.brake_threshold_straight:
                brake = 1.0

        elif target_speed <= self.brake_threshold:
                throttle = 0.0
                brake = 1.0

        self.debug = {
                # 'curve': curve,
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }

        steer, throt, brake = self.postprocess(steer, throttle, brake)

        # if(target_speed<3):
        #     print(f'*tg:{target_speed:.2f} spd:{speed:.2f} cmd:{_cmd} | steer:{steer:.2f}, throt:{throt:.2f}, brake:{brake:.2f}')
        # else:
        #     print(f'tg:{target_speed:.2f} spd:{speed:.2f} cmd:{_cmd} | steer:{steer:.2f}, throt:{throt:.2f}, brake:{brake:.2f}')

        return steer, throt, brake, target_speed

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
################################################################################


def load_state_dict(model, path):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path)

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
