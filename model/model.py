"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
import torch
from torch.nn import Module, Linear, Embedding, Parameter
from torch.nn.modules import ModuleList
from model.layers import Layer


class BQModel(Module):

    def __init__(self, dim_input_nodes, emb_size, latent_seq_len, latent_emb_size, nb_heads, dim_ff, activation_ff,
                 nb_layers, problem="tsp"):
        assert problem == "tsp" or problem == "cvrp"
        super().__init__()
        self.problem = problem

        self.begin_end_tokens = Embedding(2, emb_size)

        self.input_emb = Linear(dim_input_nodes, emb_size)
        self.encoder = Layer([emb_size, latent_emb_size], nb_heads, dim_ff, activation_ff)
        self.layers = ModuleList([Layer(latent_emb_size, nb_heads, dim_ff, activation_ff)
                                  for _ in range(nb_layers)])
        self.decoder = Layer([latent_emb_size, emb_size], nb_heads, dim_ff, activation_ff)

        output_dim = 2 if problem == "cvrp" else 1
        self.scores_projection = Linear(emb_size, output_dim)

        # latent array
        self.latent_array = Parameter(torch.FloatTensor(1, latent_seq_len, latent_emb_size))
        torch.nn.init.xavier_uniform_(self.latent_array)

    def forward(self, inputs, **problem_data):
        # inputs [batch_size, seq_len, (x_coord, y_coord, capacity, demand)]
        input_emb = self.input_emb(inputs)  # [batch_size, seq_len, emb_size]

        if self.problem != "kp":
            input_emb[:, 0, :] += self.begin_end_tokens(torch.tensor([[0]], device=inputs.device)).squeeze(1)
            input_emb[:, -1, :] += self.begin_end_tokens(torch.tensor([[1]], device=inputs.device)).squeeze(1)

        state = self.encoder(input_emb, self.latent_array.repeat(input_emb.shape[0], 1, 1))

        for layer in self.layers:
            state = layer(state, state)  # [batch_size, seq_len, emb_size]

        state = self.decoder(state, input_emb)

        scores = self.scores_projection(state).squeeze(-1) # [batch_size, seq_len]

        if self.problem == "tsp":
            # mask origin and destination
            scores[:, 0] = scores[:, -1] = -np.inf
        elif self.problem == "cvrp":
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges (direct edges to nodes with capacity larger than available demand)
            scores[..., 0][problem_data["demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf

        return scores.reshape(scores.shape[0], -1)