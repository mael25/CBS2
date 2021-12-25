import time
import argparse

from pathlib import Path

from natsort import natsorted
from torchviz import make_dot
from torchvision import transforms
import numpy as np
import torch
import tqdm

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass

import utils.bz_utils as bzu

from models.birdview import BirdViewPolicyModelSS
from utils.train_util import one_hot
from utils.datasets.birdview_lmdb import get_birdview as load_data

# Maybe experiment with this eventually...
BACKBONE = 'resnet18'
GAP = 5
N_STEP = 5
SAVE_EPOCHS = range(0, 1000, 2)
# SAVE_EPOCHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]


class LocationLoss(torch.nn.Module):
    def __init__(self, w=192, h=192, choice='l2'):
        super(LocationLoss, self).__init__()

        # IMPORTANT(bradyz): loss per sample.
        if choice == 'l1':
            self.loss = lambda a, b: torch.mean(torch.abs(a - b), dim=(1, 2))
        elif choice == 'l2':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplemented("Unknown loss: %s" % choice)

        self.img_size = torch.FloatTensor([w, h]).cuda()

    def forward(self, pred_location, gt_location):
        '''
        Note that ground-truth location is [0,img_size]
        and pred_location is [-1,1]
        '''
        gt_location = gt_location / (0.5 * self.img_size) - 1.0

        return self.loss(pred_location, gt_location)


def _log_visuals(birdview, speed, command, loss, locations, _locations, wp_method,
                 size=16):
    import cv2
    import numpy as np
    import utils.carla_utils as cu

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = list()

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rows = [x * (canvas.shape[0] // 10) for x in range(10 + 1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10 + 1)]

        def _write(text, i, j):
            cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def _dot(i, j, color, radius=2):
            x, y = int(j), int(i)
            canvas[x - radius:x + radius + 1,
            y - radius:y + radius + 1] = color

        _command = {
            1: 'LEFT', 2: 'RIGHT',
            3: 'STRAIGHT', 4: 'FOLLOW'}.get(
            torch.argmax(command[i]).item() + 1, '???')

        _wp_method = {
            0: 'OK', 1: 'Interp', 2: '<2',
            3: 'TL Stop', 4: 'Obs Stop'}.get(
            torch.argmax(wp_method[i]).item(), '???')

        _dot(0, 0, WHITE)

        for x, y in locations[i]: _dot(x, y, BLUE)
        scaling_f = 0.5 * torch.from_numpy(np.array([192, 80], dtype=np.float32))
        scaling_f = scaling_f.to(config['device'])
        for x, y in ((_locations[i] + 1) * scaling_f): _dot(x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)
        _write('Wp: %s' % _wp_method, 3, 0)
        _write('Spd: %.2f' % speed[0], 4, 0)

        images.append((loss[i].item(), canvas))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]


def train_or_eval(criterion, net, data, optim, is_train, schedular, config,
                  is_first_epoch):
    if is_train:
        desc = 'Train'
        net.train()
    else:
        desc = 'Val'
        net.eval()

    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)

    tick = time.time()
    epoch_loss = []
    for i, (cheat, location, command, speed, wp_method) in iterator:
        # if parsed.segmentation:
        #     del birdview
        #     birdview = segmentation.float()

        cheat = cheat.float().to(config['device'])
        command = one_hot(command).to(config['device'])
        speed = speed.float().to(config['device'])
        location = location.float().to(config['device'])

        pred_location = net(cheat, speed, command)
        # torch.onnx.export(net, (cheat, speed, command), 'rnn_seg_test.onnx',
        #                   input_names=['Cheat', 'Speed', 'Command'],
        #                   output_names=['Predicted_location'])

        loss = criterion(pred_location, location)
        loss_mean = loss.mean()  # Batch mean loss

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()
        epoch_loss.append(loss_mean.item())

        should_log = False
        should_log |= i % config['log_iterations'] == 0
        should_log |= not is_train
        should_log |= is_first_epoch

        if should_log:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = _log_visuals(
                cheat, speed, command, loss,
                location, pred_location, wp_method)

            bzu.log.scalar(is_train=is_train, loss_mean=loss_mean.item())
            bzu.log.image(is_train=is_train, birdview=images)

        bzu.log.scalar(is_train=is_train, fps=1.0 / (time.time() - tick))

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break

    if not is_train and not is_first_epoch and config['optimizer_args']['dynamic']:
        schedular.step(np.mean(epoch_loss))


def train(config):
    bzu.log.init(config['log_dir'])
    bzu.log.save_config(config)

    data_train, data_val = load_data(**config['data_args'])
    criterion = LocationLoss(w=config['width'], h=config['height'],choice='l1')
    net = BirdViewPolicyModelSS(config['model_args']['backbone'], config[
        'model_args']['input_channel'], seg=parsed.segmentation).to(config['device'])

    start_epoch = 0
    if config['resume']:
        log_dir = Path(config['log_dir'])
        checkpoints = sorted(log_dir.glob('model-*.th'), key=lambda path: int(path.stem.rsplit("model-", 1)[1]))
        start_epoch = int(checkpoints[-1].stem.rsplit("model-", 1)[1])
        bzu.log.epoch = start_epoch
        checkpoint = str(checkpoints[-1])
        print("load %s" % checkpoint)
        net.load_state_dict(torch.load(checkpoint))

    optim = torch.optim.Adam(net.parameters(),
                             lr=config['optimizer_args']['lr'])
    schedular = None
    if config['optimizer_args']['dynamic']:
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                               patience=config['optimizer_args']['patience'],
                                                               verbose=True)

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + int(config['max_epoch']) + 1),desc='Epoch'):
        train_or_eval(criterion, net, data_train, optim, True, schedular,config,
                      epoch == 0)
        train_or_eval(criterion, net, data_val, None, False, schedular, config,
                      epoch == 0)

        if epoch in SAVE_EPOCHS:
            torch.save(
                net.state_dict(),
                str(Path(config['log_dir']) / ('model-%d.th' % epoch)))

        bzu.log.end_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=1000)
    parser.add_argument('--max_epoch', default=1000)

    # Dataset.
    parser.add_argument('--dataset_dir',
                        default='/raid0/dian/carla_0.9.6_data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--x_jitter', type=int, default=15)
    parser.add_argument('--y_jitter', type=int, default=5)
    parser.add_argument('--scale', type=float, default=1.15)
    parser.add_argument('--angle_jitter', type=int, default=5)
    parser.add_argument('--gap', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--cmd-biased', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--segmentation', action='store_true')
    parser.add_argument('--dynamic', action='store_true', default=False)
    parser.add_argument('--combine_seg', action='store_true', default=False)

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-3)
    parsed = parser.parse_args()

    config = {
        'log_dir': parsed.log_dir,
        'resume': parsed.resume,
        'log_iterations': parsed.log_iterations,
        'max_epoch': parsed.max_epoch,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'optimizer_args': {'lr': parsed.lr, 'dynamic': parsed.dynamic,
                           'patience': 50},
        'width': 192,
        'height': 80 if parsed.segmentation else 192,
        'data_args': {
            'dataset_dir': parsed.dataset_dir,
            'batch_size': parsed.batch_size,
            'n_step': N_STEP,
            'gap': GAP,
            'crop_x_jitter': parsed.x_jitter,
            'crop_y_jitter': parsed.y_jitter,
            'angle_jitter': parsed.angle_jitter,
            'max_frames': parsed.max_frames,
            'cmd_biased': parsed.cmd_biased,
            'combine_seg': parsed.combine_seg,
            'segmentation': parsed.segmentation,
            'scale': parsed.scale,
        },
        'model_args': {
            'model': 'segmentation_thomas' if parsed.segmentation else 'birdview_dian',
            'input_channel': 5,
            'backbone': BACKBONE,
        },
    }

    train(config)
