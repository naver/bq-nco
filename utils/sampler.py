"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
from torch.utils.data import Sampler
import random

class SamplerVariousSolutionLens(Sampler):
    """
    Sampling for datasets with solution with different soulution site (KP, OP, MVP...)
    Dataset is created by chunks of data with same solution lens and then each chunk is shuffled separately
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        indices = list()
        minlen, maxlen = data_source.solution_lengths.min(), data_source.solution_lengths.max()
        for sol_len in range(minlen, maxlen):
            idxs = np.nonzero(data_source.solution_lengths == sol_len)[0]
            if len(idxs) > 0:
                indices.append(idxs)
        self.indices = indices

    def __iter__(self):
        # Return an iterator that iterates over the indices of the dataset, but keep data with same solution lengths
        # in the chunks (next to each other).
        random.shuffle(self.indices)
        shuffled_indices = list()
        for el in self.indices:
            random.shuffle(el)
            shuffled_indices.extend(el)
        return iter(shuffled_indices)

    def __len__(self):
        # Return the length of the sampler (i.e., the number of samples)
        return len(self.data_source)
