import os
import torch
import numpy as np
import ray
import tqdm
import math

from .dataset import CBSDataset, data_loader
from .logger import Logger
from .cbs import CBS

def main(args):

    cbs = CBS(args)
    # Load cbs model weights
    cbs.bev_model.load_state_dict(torch.load(cbs.bev_model_dir))
    cbs.bev_model.eval()

    logger = Logger(args)
    dataset = CBSDataset(cbs.main_data_dir, args.config_path)

    data = data_loader(dataset, args.batch_size)

    global_it = 0
    for epoch in range(args.num_epochs):
        for rgbs, lbls, sems, dlocs, spds, cmds, sem_channels_tls, locs_sem in tqdm.tqdm(data, desc=f'Epoch {epoch}'):
            info = cbs.train(rgbs, lbls, sems, dlocs, spds, cmds, sem_channels_tls, locs_sem, train='rgb')
            global_it += 1

            if global_it % args.num_iters_per_log == 0:
                logger.log_rgb(global_it, rgbs, lbls, info)

        # Save model
        torch.save(cbs.rgb_model.state_dict(), os.path.join(logger.log_dir, 'rgb_model_{}.th'.format(epoch+1)))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='carla_cbs_rgb')
    #parser.add_argument('--config-path', default='experiments/config_nocrash_cbs.yaml')
    parser.add_argument('--config-path', default='cbs/config_cbs_test.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')

    # Training data config
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-epochs', type=int, default=20)

    # Logging config
    parser.add_argument('--num-iters-per-log', type=int, default=100)

    args = parser.parse_args()

    main(args)
