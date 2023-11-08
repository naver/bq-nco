"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
from args import add_common_args, add_common_training_args
from learning.cvrp.data_iterator import DataIterator
from learning.cvrp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_cvrp')
    add_common_args(parser)
    add_common_training_args(parser)
    args = parser.parse_args()

    net, module, device, optimizer, checkpointer, other = setup_exp(args, problem="cvrp")

    # Set or re-set iteration counters and other variables from checkpoint reload
    epoch_done = 0 if other is None else other['epoch_done']
    best_current_val_metric = float('inf') if other is None else other['best_current_val_metric']

    data_iterator = DataIterator(args)

    traj_learner = TrajectoryLearner(
        args, net, module, device, data_iterator, optimizer, checkpointer)
    traj_learner.train()
