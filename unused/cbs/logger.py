import cv2
import ray
import numpy as np
import wandb
import matplotlib.pyplot as plt
from utils.visualization import visualize_birdview, visualize_semantic_processed
from matplotlib.patches import Circle

class Logger:
    def __init__(self, config):
        wandb.init(project=config.project, config=config)

    @property
    def log_dir(self):
        return wandb.run.dir

    def log_bev(self, it, lbls, info, num_log=16):

        lbls = lbls[:num_log].numpy()
        bevs = np.stack([visualize_birdview(lbl, num_channels=6) for lbl in lbls], 0)

        cmds = info.pop('cmds')
        locs = info.pop('locs')
        pred_locs = info.pop('pred_locs')

        # Draw on bevs
        for loc, pred_loc, bev, cmd in zip(locs, pred_locs, bevs, cmds):
            for t in range(loc.shape[0]):
                gx, gy = loc[t].astype(int)
                px, py = pred_loc[cmd,t].astype(int)
                cv2.circle(bev, (gx, gy), 1, (0,0,255), -1)
                cv2.circle(bev, (px, py), 1, (255,0,0), -1)


        info.update({'it': it, 'visuals': [wandb.Image(bev) for bev in bevs]})
        wandb.log(info)

    def log_sem(self, it, rgbs, lbls, info): #copy of log_rgb to be used for teacher

        rgb = rgbs[0].numpy()
        lbl = lbls[0].numpy()

        cmd = info.pop('cmds')[0]
        print(f'cmd: {cmd}')
        #tgt_rgb_loc = info.pop('locs_sem')[0]
        tgt_rgb_loc = info.pop('loc_sem')[0]
        tgt_bev_loc = info.pop('locs')[0]
        pred_rgb_loc = info.pop('pred_locs_sem')[0][cmd]
        pred_bev_loc = info.pop('pred_locs')[0][cmd]
        #pred_sem = visualize_semantic_processed(info.pop('pred_sems')[0])
        #tgt_sem = visualize_semantic_processed(info.pop('tgt_sems')[0])

        # f, [sem_axes, rgb_axes, lbl_axes] = plt.subplots(3,tgt_rgb_loc.shape[0], figsize=(30,15))
        f, [rgb_ax, lbl_ax] = plt.subplots(2,1, figsize=(30,15))
        #
        # sem_axes[0].imshow(tgt_sem)
        # sem_axes[1].imshow(pred_sem)

        rgb_ax.imshow(rgb)
        lbl_ax.imshow(visualize_birdview(lbl))
        for i in range(tgt_rgb_loc.shape[0]):
            rgb_ax.add_patch(Circle(tgt_rgb_loc[i], 4, color='blue'))
            rgb_ax.add_patch(Circle(pred_rgb_loc[i], 4, color='red'))
            lbl_ax.add_patch(Circle(tgt_bev_loc[i], 1, color='blue'))
            lbl_ax.add_patch(Circle(pred_bev_loc[i], 1, color='red'))
        info.update({'global_it': it, 'visuals': plt})
        wandb.log(info)
        plt.close('all')



    def log_locs(self, it, rgbs, lbls, info): #copy of log_rgb to be used for teacher

        rgb = rgbs[0].numpy()
        cmd = info.pop('cmds')[0]

        locs_image_space = info.pop('locs_image_space')[0]
        #locs_sem_off = info.pop('locs_sem_off')[0]
        pred_locs_image_space = info.pop('pred_locs_image_space')[0][cmd]

        f, ax = plt.subplots(figsize=(30,15))
        ax.imshow(rgb)
        # print(f'GT WoR: {locs_sem_off[0]}')
        # print(f'GT CBS: {locs_sem_cbs[0]}')
        # print(f'Pred  : {pred_locs_sem[0]}')
        #for i in range(locs_sem_off.shape[0]):
            #ax.add_patch(Circle(locs_sem_off[i], 4, color='blue'))
        for i in range(locs_image_space.shape[0]):
            ax.add_patch(Circle(locs_image_space[i], 4, color='red'))
        for i in range(pred_locs_image_space.shape[0]):
            ax.add_patch(Circle(pred_locs_image_space[i], 4, color='yellow'))
        info.update({'global_it': it, 'visuals': plt})
        wandb.log(info)
        plt.close('all')


    def log_rgb(self, it, rgbs, lbls, info):

        rgb = rgbs[0].numpy()
        lbl = lbls[0].numpy()

        cmd = info.pop('cmds')[0]
        tgt_rgb_loc = info.pop('tgt_rgb_locs')[0]
        tgt_bev_loc = info.pop('tgt_bev_locs')[0]
        pred_rgb_loc = info.pop('pred_rgb_locs')[0]
        pred_bev_loc = info.pop('pred_bev_locs')[0]
        pred_sem = visualize_semantic_processed(info.pop('pred_sems')[0])
        tgt_sem = visualize_semantic_processed(info.pop('tgt_sems')[0])

        f, [sem_axes, rgb_axes, lbl_axes] = plt.subplots(3,tgt_rgb_loc.shape[0], figsize=(30,15))

        sem_axes[0].imshow(tgt_sem)
        sem_axes[1].imshow(pred_sem)

        for i in range(tgt_rgb_loc.shape[0]):
            rgb_ax = rgb_axes[i]
            lbl_ax = lbl_axes[i]

            rgb_ax.imshow(rgb)
            lbl_ax.imshow(visualize_birdview(lbl))
            rgb_ax.set_title({0:'Left',1:'Right',2:'Straight',3:'Follow'}.get(cmd,'???'))
            for tgt_rgb, tgt_bev, pred_rgb, pred_bev in zip(tgt_rgb_loc[i],tgt_bev_loc[i],pred_rgb_loc[i],pred_bev_loc[i]):

                rgb_ax.add_patch(Circle(tgt_rgb, 4, color='blue'))
                rgb_ax.add_patch(Circle(pred_rgb, 4, color='red'))

                lbl_ax.add_patch(Circle(tgt_bev, 1, color='blue'))
                lbl_ax.add_patch(Circle(pred_bev, 1, color='red'))

        info.update({'global_it': it, 'visuals': plt})
        wandb.log(info)
        plt.close('all')
