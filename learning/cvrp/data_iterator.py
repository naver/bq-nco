"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from argparse import Namespace
from learning.cvrp.dataloading.dataset import load_dataset

class DataIterator:

    def __init__(self, args: Namespace):

        if args.train_dataset is not None:
            self.train_trajectories = load_dataset(args.train_dataset, args.train_batch_size, True, "train")

        if args.val_dataset is not None:
            self.val_trajectories = load_dataset(args.val_dataset, args.val_batch_size, False, "val")

        self.test_trajectories = load_dataset(args.test_dataset, args.test_batch_size, False, "test")
