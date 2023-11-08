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

    def __init__(self, node_coords, tour_lens=None):
        self.node_coords = node_coords
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        dist_matrix = squareform(pdist(node_coords, metric='euclidean'))

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix)
        item_dict.nodes_coord = torch.Tensor(node_coords)
        if self.tour_lens is not None:
            item_dict.tour_len = self.tour_lens[item]
        else:
            item_dict.tour_len = torch.Tensor([])

        return item_dict


def load_dataset(filename, batch_size, shuffle=False, what="test"):
    data = np.load(filename)

    if what == "train":
        assert data["reorder"]

    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample_suffix if what == "train" else None

    dataset = DataLoader(DataSet(data["coords"], tour_lens=tour_lens), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset


def collate_func_with_sample_suffix(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-TSP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].nodes_coord)
    subproblem_size = np.random.randint(4, nb_nodes + 1)
    begin_idx = nb_nodes + 1 - subproblem_size
    l_dataset_items_new = prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size)
    return default_collate(l_dataset_items_new)

def prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size):
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if type(v) == numpy.float64:
                v_ = 0.
            elif len(v.shape) == 1 or k == 'nodes_coord':
                v_ = v[begin_idx:begin_idx+subproblem_size, ...]
            else:
                v_ = v[begin_idx:begin_idx+subproblem_size, begin_idx:begin_idx+subproblem_size]
            d_new.update({k+'_s': v_})
        l_dataset_items_new.append({**d, **d_new})
    return l_dataset_items_new


def sample_subproblem(nb_nodes):
    subproblem_size = np.random.randint(4, nb_nodes + 1)  # between _ included and nb_nodes + 1 excluded
    begin_idx = np.random.randint(nb_nodes - subproblem_size + 1)
    return begin_idx, subproblem_size

