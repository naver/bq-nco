"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch.nn import Module

class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self._reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': (lambda x: x)}[reduction]

    def forward(self, scores, target):
        return self._reduction(torch.logsumexp(scores, dim=-1) - torch.sum(torch.nan_to_num(scores*target), dim=-1))
