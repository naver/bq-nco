"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
import torch
from torch.nn import Module, Linear, Embedding
from torch.nn.modules import ModuleList
from model.encoder import EncoderLayer


class BQModel(Module):

    def __init__(self, dim_input_nodes, emb_size, dim_ff, activation_ff, nb_layers_encoder, nb_heads,
                 activation_attention, dropout, batchnorm, problem="tsp"):
        assert problem == "tsp" or problem == "cvrp" or problem == "kp" or problem == "op"
        super().__init__()
        self.problem = problem
        self.input_emb = Linear(dim_input_nodes, emb_size)
        if problem != "kp":
            self.begin_end_tokens = Embedding(2, emb_size)

        self.nb_layers_encoder = nb_layers_encoder
        self.encoder = ModuleList([EncoderLayer(nb_heads, activation_attention, emb_size, dim_ff, activation_ff,
                                                dropout, batchnorm) for _ in range(nb_layers_encoder)])

        if problem == "cvrp":
            output_dim = 2
        else:
            output_dim = 1
        self.scores_projection = Linear(emb_size, output_dim)

    def forward(self, inputs, **problem_data):
        # inputs
        #     TSP [batch_size, seq_len, (x_coord, y_coord)]
        #    CVRP [batch_size, seq_len, (x_coord, y_coord, demand, current_capacity)]
        #      OP [batch_size, seq_len, (x_coord, y_coord, node_value, upper_bound)]
        #      KP [batch_size, seq_len, (weight, value, remaining_capacity)]
        if self.problem == "op":
            assert "dist_matrices" in problem_data

        input_emb = self.input_emb(inputs)  # [batch_size, seq_len, emb_size]
        if self.problem != "kp":
            input_emb[:, 0, :] += self.begin_end_tokens(torch.tensor([[0]], device=inputs.device)).squeeze(1)
            input_emb[:, -1, :] += self.begin_end_tokens(torch.tensor([[1]], device=inputs.device)).squeeze(1)

        state = input_emb
        for layer in self.encoder:
            state = layer(state)  # [batch_size, seq_len, emb_size]

        scores = self.scores_projection(state).squeeze(-1) # [batch_size, seq_len]

        if self.problem == "tsp":
            # mask origin and destination
            scores[:, 0] = scores[:, -1] = -np.inf
        elif self.problem == "cvrp":
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges (direct edges to nodes with capacity larger than available demand)
            scores[..., 0][problem_data["demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
        elif self.problem == "op":
            scores[:, 0] = scores[:, -1] = -np.inf
            # op - mask all nodes with cost to go there and back to depot > current upperbound
            # todo: update with real values
            scores[problem_data["dist_matrices"][:, 0] +
                   problem_data["dist_matrices"][:, -1] - inputs[..., 3] > 1e-6] = -np.inf
        elif self.problem == "kp":
            # kp - mask all nodes with weights > current capacity
            scores[problem_data["weights"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf

        return scores.reshape(scores.shape[0], -1)
