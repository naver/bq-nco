"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from argparse import Namespace
from learning.tsp.dataloading.dataset import load_dataset

class DataIterator:

    def __init__(self, args: Namespace):
        if args.train_dataset is not None:
            # we have a training
            self.train_trajectories = load_dataset(args.train_dataset, args.train_batch_size,True,
                                                   "train")

        if args.val_dataset is not None:
            self.val_trajectories = load_dataset(args.val_dataset, args.val_batch_size, False, "val")

        self.test_trajectories = load_dataset(args.test_dataset, args.test_batch_size, False, "test")



class TSPLibTestDataIterator:

    def __init__(self, filename):
        self.test_trajectories = dict()
        name = filename.split("/")[-1].split(".")[0]
        self.test_trajectories[name] = load_dataset(filename, 1, False, True, "test")
