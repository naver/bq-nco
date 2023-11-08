"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import subprocess
import random
import numpy as np
import torch
from torch import Tensor, nn


def get_params_to_log(args):
    args.update(
        {'commit_id': subprocess.run(
            ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]}
    )
    return args


def do_lr_decay(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        new_learning_rate = param_group['lr'] * decay_rate
        param_group['lr'] = new_learning_rate
    print("Learning rate decayed by {:.4f}".format(decay_rate))


def set_seed(seed: int):
    if seed is None:
        seed = random.randint(0, 1e5)

    random.seed(seed)
    np.random.seed(seed)  # CAREFUL if doing sampling inside dataloaders!
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)


def print_model_params_info(module, detailed=False):
    sum = 0
    for name, values in module.state_dict().items():
        dim = 1
        if detailed:
            print(name, ":", values.shape)
        for val in values.shape:
            dim *= val
        sum += dim
    print("Total number of parameters:", sum)


class EpochMetrics:
    # dict of metrics values over epoch
    # makes sure the same metric names are given for each update

    def __init__(self):
        self.metrics = None

    def update(self, d):
        d = {k: (v.item() if isinstance(v, Tensor) else v) for k, v in d.items()}
        if self.metrics is None:
            self.metrics = {kd: [vd] for kd, vd in d.items()}
        else:
            for (k, v), (kd, vd) in zip(self.metrics.items(), d.items()):
                assert k == kd
                v.append(vd)

    def get_means(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}


def get_opt_gap(predicted_tour_lens, gt_tour_lens):
    return 100 * ((predicted_tour_lens - gt_tour_lens) / gt_tour_lens).mean().item()


def maybe_cuda_ddp(net):
    # Rem: DataParallel works fine on slurm servers but non on ssh-debug (chaos) servers,
    # debug with one gpu in CUDA_VISIBLE_DEVICES
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    module = net.module if isinstance(net, nn.DataParallel) else net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    return net, module, device

def get_lr(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        if lr is not None:
            assert param_group['lr'] == lr, "verif not implemented for different lr per param group. "
        lr = param_group['lr']
    return lr


def compute_tour_lens(paths: Tensor, adj_matrices: Tensor) -> Tensor:
    batch_idx = torch.arange(len(paths))
    distances = torch.sum(torch.stack([adj_matrices[batch_idx, paths[batch_idx, idx], paths[batch_idx, idx + 1]]
                                       for idx in range(paths.shape[1] - 1)]).transpose(0, 1), axis=1)
    distances += adj_matrices[batch_idx, paths[batch_idx, 0], paths[batch_idx, -1]]
    return distances
