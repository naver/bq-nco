"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, demands, capacities, remaining_capacities, tour_lens=None, via_depots=None):
        self.node_coords = node_coords
        self.demands = demands
        self.capacities = capacities
        self.remaining_capacities = remaining_capacities
        self.via_depots = via_depots
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        demands = self.demands[item]
        capacity = self.capacities[item]
        if self.tour_lens is not None:
            tour_len = self.tour_lens[item]
        else:
            tour_len = numpy.array([])

        if self.remaining_capacities is not None:
            via_depots = self.via_depots[item]
            current_capacities = self.remaining_capacities[item]
        else:
            via_depots = numpy.array([])
            current_capacities = numpy.array([])

        distance_matrix = squareform(pdist(node_coords, metric='euclidean'))

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(distance_matrix)
        item_dict.node_coords = torch.Tensor(node_coords)
        item_dict.demands = torch.Tensor(demands)
        item_dict.capacities = torch.tensor(capacity).float()
        item_dict.remaining_capacities = torch.Tensor(current_capacities)
        item_dict.tour_len = torch.tensor(tour_len)
        item_dict.via_depots = torch.Tensor(via_depots).long()
        return item_dict


def load_dataset(filename, batch_size, shuffle=False, what="test"):
    data = np.load(filename)

    if what == "train":
        assert data["reorder"]

    node_coords = data["coords"]
    demands = data["demands"]
    capacities = data["capacities"]


    # in training dataset we have via_depots and remaining capacities but not tour lens
    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
    remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
    via_depots = data["via_depots"] if "via_depots" in data.keys() else None

    collate_fn = collate_func_with_sample if what == "train" else None

    dataset = DataLoader(DataSet(node_coords, demands, capacities,
                                 remaining_capacities=remaining_capacities,
                                 tour_lens=tour_lens,
                                 via_depots=via_depots), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset


def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-CVRP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    begin_idx = np.random.randint(0, nb_nodes - 3)  # between _ included and nb_nodes + 1 excluded

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if k == "dist_matrices":
                v_ = v[begin_idx:, begin_idx:]
            elif k == "remaining_capacities":
                v_ = v[begin_idx]
            elif k == "capacities":
                v_ = v
            else:
                v_ = v[begin_idx:, ...]

            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)

