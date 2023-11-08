"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
from args import add_common_args
from learning.op.data_iterator import DataIterator
from learning.op.traj_learner import TrajectoryLearner
from utils.exp import setup_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test_op')
    add_common_args(parser)  # (only need common args)
    args = parser.parse_args()

    net, module, device, _, checkpointer, _ = setup_exp(args, problem="op", is_test=True)

    data_iterator = DataIterator(args)

    traj_learner = TrajectoryLearner(args, net, module, device, data_iterator, checkpointer=checkpointer)
    # (for eval, no need for optimizer, watcher, checkpointer)

    start_time = time.time()
    traj_learner.val_test()
    print(f"Inference time {(time.time() - start_time):.3f}s")
