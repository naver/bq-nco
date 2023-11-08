"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import entmax
import numpy as np
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, use_biases=True, scale_dot_att=False, activation="softmax"):
        super().__init__()
        assert (dim % num_heads == 0)

        self.emb_size = dim
        self.scale_dot_att = scale_dot_att
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert activation == "softmax" or activation == "entmax"
        self.activation = activation

        self.lambda1 = nn.Parameter(torch.FloatTensor(num_heads, dim, self.head_dim))
        self.lambda2 = nn.Parameter(torch.FloatTensor(num_heads, dim, self.head_dim))
        self.theta1 = nn.Parameter(torch.FloatTensor(num_heads, dim, self.head_dim))
        self.theta2 = nn.Parameter(torch.FloatTensor(num_heads, dim, self.head_dim))

        self.bias_lambda = torch.nn.Parameter(torch.FloatTensor(num_heads, self.head_dim)) if use_biases else None
        self.bias_theta = torch.nn.Parameter(torch.FloatTensor(num_heads, self.head_dim)) if use_biases else None

        self._reset_params()

    def _reset_params(self):
        torch.nn.init.xavier_uniform_(self.lambda1)
        torch.nn.init.xavier_uniform_(self.lambda2)
        torch.nn.init.xavier_uniform_(self.theta1)
        torch.nn.init.xavier_uniform_(self.theta2)
        torch.nn.init.zeros_(self.bias_lambda)
        torch.nn.init.zeros_(self.bias_theta)

    def forward(self, x, y, mask=None):
        r = torch.einsum('bmp,kpd->bmkd', x, self.lambda1)  # r: batch_size, seq_len, num_heads, head_size
        if self.bias_lambda is not None:
            r = r + self.bias_lambda
        dot_att = torch.einsum('bmkd,kqd,bnq->kbmn', r, self.lambda2, y)
        if self.scale_dot_att:
            dot_att *= self.head_dim ** -0.5

        if mask is not None:
            if len(mask.shape) == 2:
                # repeat over num_heads and batch_size
                mask = mask.repeat(dot_att.shape[0], dot_att.shape[1], 1, 1)
            else:
                # repeat over num_heads
                mask = mask.repeat(dot_att.shape[0], 1, 1, 1)
            dot_att[mask == 1] = -np.inf

        if self.activation == "softmax":
            att_weights = torch.softmax(dot_att, dim=-2)
        else:
            att_weights = entmax.entmax15(dot_att, dim=-2)

        r = torch.einsum('kbmn,bmp,kpd->bnkd', att_weights, x, self.theta1) # r: batch_size, seq_len, num_heads, head_size
        if self.bias_theta is not None:
            r = r + self.bias_theta
        output = torch.einsum('bnkd,kqd->bnq', r, self.theta2)

        return output
