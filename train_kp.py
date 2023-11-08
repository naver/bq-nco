"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
from args import add_common_args, add_common_training_args
from learning.kp.data_iterator import DataIterator
from learning.kp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_kp')
    add_common_args(parser)
    add_common_training_args(parser)
    args = parser.parse_args()

    net, module, device, optimizer, checkpointer, other = setup_exp(args, problem="kp")

    data_iterator = DataIterator(args)

    traj_learner = TrajectoryLearner(args, net, module, device, data_iterator, optimizer, checkpointer)
    traj_learner.train()