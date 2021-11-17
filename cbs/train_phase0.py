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
    logger = Logger(args)
    dataset = CBSDataset(cbs.main_data_dir, args.config_path, jitter=True)

    data = data_loader(dataset, args.batch_size)

    global_it = 0
    for epoch in range(args.num_epochs):
        for rgbs, lbls, sems, dlocs, spds, cmds, sem_channels_tls, locs_sem in tqdm.tqdm(data, desc=f'Epoch {epoch}'):
            info = cbs.train(rgbs, lbls, sems, dlocs, spds, cmds, sem_channels_tls, locs_sem, train='bev')
            global_it += 1

            if global_it % args.num_iters_per_log == 0:
                logger.log_bev(global_it, lbls, info.copy())
                #Added: log also in the segm. view:
                #logger.log_sem(global_it, rgbs, lbls, info)


        # Save model
        torch.save(cbs.bev_model.state_dict(), os.path.join(logger.log_dir, 'bev_model_{}.th'.format(epoch+1)))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='carla_cbs')
    #parser.add_argument('--config-path', default='experiments/config_nocrash_cbs.yaml')
    parser.add_argument('--config-path', default='cbs/config_cbs_test.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')

    # Training data config
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-epochs', type=int, default=20)

    # Logging config
    #parser.add_argument('--num-iters-per-log', type=int, default=100)
    parser.add_argument('--num-iters-per-log', type=int, default=1)

    args = parser.parse_args()

    main(args)
