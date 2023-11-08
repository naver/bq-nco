"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.sampler import SamplerVariousSolutionLens

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, node_values, upper_bounds, collected_rewards, solution_lengths):
        self.node_coords = node_coords
        self.node_values = node_values
        self.upper_bounds = upper_bounds
        self.collected_rewards = collected_rewards
        self.solution_lengths = solution_lengths

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]

        dist_matrix = squareform(pdist(node_coords, metric='euclidean'))

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.node_coords = torch.tensor(self.node_coords[item]).float()
        item_dict.dist_matrices = torch.tensor(dist_matrix).float()
        item_dict.node_values = torch.tensor(self.node_values[item]).float()
        item_dict.upper_bounds = torch.tensor(self.upper_bounds[item]).float()
        if self.solution_lengths is not None:
            item_dict.solution_lengths = self.solution_lengths[item]
        else:
            item_dict.solution_lengths = torch.Tensor([])
        if self.collected_rewards is not None:
            item_dict.collected_rewards = self.collected_rewards[item]
        else:
            item_dict.collected_rewards = torch.tensor([])
        return item_dict


def load_dataset(filename, batch_size, shuffle=False, what="test"):
    data = np.load(filename)

    if what == "train":
        assert data["reorder"]

    collate_fn = collate_func if what == "train" else None
    solution_lengths = data["solution_lengths"] if "solution_lengths" in data else None
    collected_rewards = data["collected_rewards"] if "collected_rewards" in data else None
    dataset = DataSet(data["coords"], data["values"], data["upper_bounds"], collected_rewards, solution_lengths)
    sampler = SamplerVariousSolutionLens(dataset) if shuffle else None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=sampler, collate_fn=collate_fn)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    sol_length = np.min([el["solution_lengths"] for el in l_dataset_items])
    num_sampled_nodes = random.randint(0, sol_length - 2)
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = dict()
        for k, v in d.items():
            if k == "node_coords" or k == "node_values":
                v_ = v[num_sampled_nodes:, ...]
            elif k == "dist_matrices":
                v_ = v[num_sampled_nodes:, num_sampled_nodes:]
            elif k == "upper_bounds":
                v_ = v - sum([d["dist_matrices"][i, i+1] for i in range(0, num_sampled_nodes)])
                assert v_ > 0
            else:
                v_ = v
            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)
