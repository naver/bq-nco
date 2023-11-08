"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.sampler import SamplerVariousSolutionLens


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, weights, values, capacities, solution_lengths, optimal_values, scale):
        self.capacities = capacities
        self.weights = weights
        self.values = values
        self.solution_lengths = solution_lengths
        self.optimal_values = optimal_values
        self.scale = scale

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.remaining_capacities = torch.tensor(self.capacities[item]).float()
        item_dict.weights = torch.tensor(self.weights[item]).float()
        item_dict.values = torch.tensor(self.values[item]).float()
        if self.solution_lengths is not None:
            item_dict.solution_lengths = self.solution_lengths[item]
        else:
            item_dict.solution_lengths = torch.tensor([])
        if self.optimal_values is not None:
            item_dict.optimal_values = self.optimal_values[item]
        else:
            item_dict.optimal_values = torch.tensor([])
        item_dict.scale = torch.tensor(self.scale).float()
        return item_dict


def load_dataset(filename, batch_size, shuffle, what):
    data = np.load(filename)
    if what == "train":
        assert data["reorder"]
    collate_fn = collate_func if what == "train" else None
    solution_lengths = data["solution_lengths"] if "solution_lengths" in data else None
    optimal_values = data["optimal_values"] if "optimal_values" in data else None

    dataset = DataSet(data["weights"], data["values"], data["capacities"], solution_lengths, optimal_values,
                      data["scale"])
    sampler = SamplerVariousSolutionLens(dataset) if shuffle else None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=sampler, collate_fn=collate_fn)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    sol_length = np.min([el["solution_lengths"] for el in l_dataset_items])
    num_samples_nodes = random.randint(0, sol_length - 1)
    l_dataset_items_new = []
    for d in l_dataset_items:
        new_item = dict()
        to_remove = np.random.choice(d["solution_lengths"], num_samples_nodes, replace=False)

        keep_filter = torch.full([len(d["weights"])], True)
        keep_filter[to_remove] = False
        new_item["weights_s"] = d["weights"][keep_filter]
        new_item["values_s"] = d["values"][keep_filter]
        new_item["remaining_capacities_s"] = d["remaining_capacities"] - torch.tensor(1.) * sum(d["weights"][~keep_filter])
        new_item["optimal_values_s"] = d["optimal_values"]
        new_item["scale"] = d["scale"]
        new_item["solution_probs_s"] = torch.full([len(new_item["weights_s"])], -np.inf)
        new_item["solution_probs_s"][:d["solution_lengths"] - num_samples_nodes] = 1.
        l_dataset_items_new.append(new_item)

    return default_collate(l_dataset_items_new)