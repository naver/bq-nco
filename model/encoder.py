"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear
from torch.nn.modules import Dropout
from torch.nn.parameter import Parameter
from model.attention import MultiHeadAttention

class ResidualNorm(nn.Module):

    def __init__(self, emb_size, dropout, batchnorm):
        super().__init__()
        self.batchnorm = batchnorm
        self.norm = BatchNorm1d(emb_size, track_running_stats=False) if batchnorm else None
        self.alpha = 1. if self.batchnorm else Parameter(torch.tensor(0.))  # 'ReZero' https://arxiv.org/abs/2003.04887)
        self.dropout = Dropout(dropout)

    def forward(self, state_before, modified_after):
        # residual connection
        state_before = state_before + self.alpha * self.dropout(modified_after)
        # batch/layer norm
        if self.batchnorm:  # expects batch_size, emb_size, seq_len
            state_before = state_before.transpose(2, 1)
            state_before = self.norm(state_before)
            return state_before.transpose(2, 1)
        else:
            return state_before


class EncoderLayer(nn.Module):
    def __init__(self, nb_heads, activation_attention, emb_size, dim_ff, activation_ff, dropout, batchnorm):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(emb_size, nb_heads, activation=activation_attention)

        self.linear1 = Linear(emb_size, dim_ff)
        self.linear2 = Linear(dim_ff, emb_size)

        assert activation_ff == "relu" or activation_ff == "gelu"

        if activation_ff == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.GELU()

        self.res_norm1 = ResidualNorm(emb_size, dropout, batchnorm)
        self.res_norm2 = ResidualNorm(emb_size, dropout, batchnorm)

    def forward(self, state):
        state_rc = state
        # self attention
        state = self.attn(state, state)
        # residual + norm
        state = self.res_norm1(state_rc, state)
        state_rc = state
        # linear
        state = self.linear2(self.activation(self.linear1(state)))
        # residual + norm
        return self.res_norm2(state_rc, state)

