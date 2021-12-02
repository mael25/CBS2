import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils import _numpy
from .models import PointModel, RGBPointModel, Converter
from .models.converter_cbs import CoordinateConverterCBS, Transform, Rotation, Location

class CBS:
    def __init__(self, args):

        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        print('\n---------------------CBS {}-----------------------\n'.format(args.config_path))
        for key, value in config.items():
            setattr(self, key, value)

        self.crop_size = 64
        self.num_cmds = 6

        # Save configs
        self.device = torch.device(args.device)
        self.T = self.num_plan # T in CBS

        # Create models
        self.bev_model = PointModel(
            'resnet18',
            #height=64, width=64,
            height=240-self.crop_top-self.crop_bottom, width=480,
            #input_channel=12,
            input_channel=5, # [4,7,8,10,18]
            output_channel=self.T*self.num_cmds
        ).to(self.device)

        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            height=240-self.crop_top-self.crop_bottom, width=480,
            output_channel=self.T*self.num_cmds,
            ppm_bins = self.ppm_bins
        ).to(self.device)

        self.bev_optim = optim.Adam(self.bev_model.parameters(), lr=args.lr)
        self.rgb_optim = optim.Adam(self.rgb_model.parameters(), lr=args.lr)

        self.converter = Converter(offset=0.0, scale=[1.0,1.0], fov=120).to(self.device)

        sensor_transform = Transform(Location(), Rotation())
        self.convertercbs = CoordinateConverterCBS(sensor_transform, fov=120)

    def train(self, rgbs, lbls, sems, locs, spds, cmds, sem_channels_tls, locs_image_space, train='image'):

        rgbs = rgbs.permute(0,3,1,2).float().to(self.device)
        lbls = lbls.permute(0,3,1,2).float().to(self.device)
        sems = sems.long().to(self.device)
        locs = locs.float().to(self.device)
        spds = spds.float().to(self.device)
        cmds = cmds.long().to(self.device)
        sem_channels_tls = sem_channels_tls.permute(0,3,1,2).float().to(self.device)
        locs_image_space = locs_image_space.float().to(self.device)

        if train == 'bev':
            #print('Train BEV (CBS)\n')
            return self.train_bev(sem_channels_tls, spds, locs, cmds, locs_image_space)

        elif train == 'rgb':
            return self.train_rgb(rgbs, sem_channels_tls, sems, spds, cmds)

        else:
            raise NotImplementedError

    def train_bev(self, sem_channels_tls, spds, locs, cmds, locs_image_space):


        pred_locs_image_space_normalized = self.bev_model(sem_channels_tls, spds).view(-1,self.num_cmds,self.T,2) # Teacher predictions in normalized image space [-1,1]x[-1,1]
        #print(np.max(_numpy(pred_locs_sem)))

        # Scale pred locs
        #pred_locs = (pred_locs+1) * self.crop_size/2
        pred_locs_image_space = (pred_locs_image_space_normalized+1) * self.rgb_model.img_size/2 # Teacher predicitons in image space [0,224]x[0,480]

        #print(pred_locs_image_space)
        print(f'WoR: {self.converter.cam_to_world(pred_locs_image_space[1,0,:])}')
        print(f'CBS: {self.convertercbs.unproject(pred_locs_image_space[1,0,:].cpu().detach())}')

        #pred_locs_bv_space = self.cam_to_bev(pred_locs_image_space)

        locs_image_space_normalized = (locs_image_space / (self.rgb_model.img_size/2)) -1 # Ground truth in normalized image space [-1,1]x[-1,1]

        #loss = F.mse_loss(pred_locs.gather(1, cmds[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1), locs)
        loss = F.mse_loss(pred_locs_image_space_normalized.gather(1, cmds[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1), locs_image_space_normalized)

        self.bev_optim.zero_grad()
        loss.backward()
        self.bev_optim.step()

        return dict(
            loss=float(loss),
            cmds=_numpy(cmds),
            locs=_numpy(locs),
            #locs_image_space_wor_conv=_numpy(self.bev_to_cam(locs)),
            locs_image_space=_numpy(locs_image_space),
            pred_locs_image_space=_numpy(pred_locs_image_space),
        )

    def train_rgb(self, rgbs, sem_channels_tls, sems, spds, cmds):
        #print('Train RGB (CBS)\n')

        with torch.no_grad():
            #tgt_bev_locs = (self.bev_model(sem_channels_tls, spds).view(-1,self.num_cmds,self.T,2)+1) * self.crop_size/2
            teacher_pred_locs_image_space_normalized = (self.bev_model(sem_channels_tls, spds).view(-1,self.num_cmds,self.T,2))
            teacher_pred_locs_image_space = (teacher_pred_locs_image_space_normalized + 1) * self.rgb_model.img_size/2
            #tgt_rgb_locs = (self.bev_model(sem_channels_tls, spds).view(-1,self.num_cmds,self.T,2)+1) * self.rgb_model.img_size/2
            # As we replaced bev by sem, we should make the transformation to bev as it corresponds to the space of locs
            #tgt_bev_locs = self.cam_to_bev(tgt_rgb_locs)

        student_pred_locs_image_space_normalized, student_pred_sems = self.rgb_model(rgbs, spds)
        student_pred_locs_image_space_normalized = (student_pred_locs_image_space_normalized.view(-1,self.num_cmds,self.T,2))
        student_pred_locs_image_space = (student_pred_locs_image_space_normalized + 1) * self.rgb_model.img_size/2
        #tgt_rgb_locs = self.bev_to_cam(tgt_bev_locs) #Not needed as sem and rgb are same orientation and format
        #pred_bev_locs = self.cam_to_bev(pred_rgb_locs)

        act_loss = F.l1_loss(student_pred_locs_image_space_normalized, teacher_pred_locs_image_space_normalized, reduction='none').mean(dim=[2,3])

        turn_loss = (act_loss[:,0]+act_loss[:,1]+act_loss[:,2]+act_loss[:,3])/4
        lane_loss = (act_loss[:,4]+act_loss[:,5]+act_loss[:,3])/3
        foll_loss = act_loss[:,3]

        is_turn = (cmds==0)|(cmds==1)|(cmds==2)
        is_lane = (cmds==4)|(cmds==5)

        loc_loss = torch.mean(torch.where(is_turn, turn_loss, foll_loss) + torch.where(is_lane, lane_loss, foll_loss))

        # multip_branch_losses = losses.mean(dim=[1,2,3])
        # single_branch_losses = losses.mean(dim=[2,3]).gather(1, cmds[:,None]).mean(dim=1)

        # loc_loss = torch.where(cmds==3, single_branch_losses, multip_branch_losses).mean()
        seg_loss = F.cross_entropy(F.interpolate(student_pred_sems,scale_factor=4), sems)

        loss = loc_loss + self.seg_weight * seg_loss

        self.rgb_optim.zero_grad()
        loss.backward()
        self.rgb_optim.step()

        return dict(
            loc_loss=float(loc_loss),
            seg_loss=float(seg_loss),
            cmds=_numpy(cmds),
            tgt_rgb_locs=_numpy(teacher_pred_locs_image_space),
            tgt_bev_locs=_numpy(self.cam_to_bev(teacher_pred_locs_image_space)),
            pred_rgb_locs=_numpy(student_pred_locs_image_space),
            pred_bev_locs=_numpy(self.cam_to_bev(student_pred_locs_image_space)),
            tgt_sems=_numpy(sems),
            pred_sems=_numpy(student_pred_sems.argmax(1)),
        )


    def bev_to_cam(self, bev_coords):

        bev_coords = bev_coords.clone()
        bev_coords[...,1] =  self.crop_size/2 - bev_coords[...,1]
        bev_coords[...,0] = -self.crop_size/2 + bev_coords[...,0]
        world_coords = torch.flip(bev_coords, [-1])

        cam_coords = self.converter.world_to_cam(world_coords)

        return cam_coords

    def cam_to_bev(self, cam_coords):
        world_coords = self.converter.cam_to_world(cam_coords)
        bev_coords = torch.flip(world_coords, [-1])
        bev_coords[...,1] *= -1

        return bev_coords + self.crop_size/2
