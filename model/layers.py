"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch import nn
from torch.nn import LayerNorm, Linear
from model.attention import MultiHeadAttention


class Layer(nn.Module):
    def __init__(self, emb_sizes, nb_heads, dim_ff, activation_ff):
        super(Layer, self).__init__()
        if type(emb_sizes) == int:
            emb_size_x = emb_size_y = emb_sizes
        else:
            emb_size_x, emb_size_y = emb_sizes

        self.attn = MultiHeadAttention(emb_sizes, num_heads=nb_heads)
        assert activation_ff == "relu" or activation_ff == "gelu"

        if activation_ff == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.GELU()

        self.layer_norm1 = LayerNorm(emb_size_x)
        self.layer_norm2 = LayerNorm(emb_size_y)
        self.layer_norm3 = LayerNorm(emb_size_y)
        self.linear1 = Linear(emb_size_y, dim_ff)
        self.linear2 = Linear(dim_ff, emb_size_y)

    def forward(self, x, y):
        state_rc = y
        # self attention
        state = self.attn(self.layer_norm1(x), self.layer_norm2(y))
        # residual + norm
        state = state + state_rc
        state_rc = state
        # linear
        state = self.linear2(self.activation(self.linear1(self.layer_norm3(state))))
        # residual + norm
        return state_rc + state