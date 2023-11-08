"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""


import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


@dataclass
class DecodingSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    node_coords: Tensor
    distance_matrices: Tensor
    demands: Tensor
    current_capacities: Tensor
    original_idxs: Tensor


def reconstruct_tours(paths: Tensor, via_depots: Tensor):
    bs = paths.shape[0]
    complete_paths = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_to_add = paths[:, pos].tolist()
        for instance in (via_depots[:, pos] == True).nonzero().squeeze(-1).cpu().numpy():
            complete_paths[instance].append(0)
        for instance in range(bs):
            complete_paths[instance].append(nodes_to_add[instance])

    return complete_paths


def decode(node_coords: Tensor, adj_matrices: Tensor, demands: Tensor, capacities: Tensor, net: Module,
           beam_size: int, knns: int, make_tours: bool = False) -> Tensor:
    if beam_size == 1:
        paths, via_depots, tour_lengths = greedy_decoding_loop(node_coords, adj_matrices, demands, capacities, net,
                                                               knns)
    else:
        paths, via_depots, tour_lengths = beam_search_decoding_loop(node_coords, adj_matrices, demands, capacities, net,
                                                                    beam_size, knns)
    num_nodes = node_coords.shape[1]
    assert paths.sum(axis=1).sum() == paths.shape[0] * .5 * (num_nodes - 1) * num_nodes

    if make_tours:
        tours = reconstruct_tours(paths, via_depots)
    else:
        tours = None

    return tour_lengths, tours


def greedy_decoding_loop(node_coords: Tensor, adj_matrices: Tensor, demands: Tensor, capacities: Tensor,
                         net: Module, knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=node_coords.device)
    initial_capacities = copy.deepcopy(capacities).unsqueeze(-1)
    paths[:, -1] = num_nodes - 1
    lenghts = torch.zeros(bs, device=node_coords.device)
    sub_problem = DecodingSubPb(node_coords, adj_matrices, demands, initial_capacities, original_idxs)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, sub_problem = greedy_decoding_step(capacities, sub_problem, net, knns)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot

        # compute lenghts for direct edges
        lenghts[~via_depots[:, dec_pos]] += adj_matrices[~via_depots[:, dec_pos],
                                                         paths[~via_depots[:, dec_pos], dec_pos-1],
                                                         paths[~via_depots[:, dec_pos], dec_pos]]
        # compute lenghts for edges via depot
        lenghts[via_depots[:, dec_pos]] += adj_matrices[via_depots[:, dec_pos],
                                                        paths[via_depots[:, dec_pos], dec_pos-1],
                                                        paths[via_depots[:, dec_pos], 0]] +\
                                           adj_matrices[via_depots[:, dec_pos],
                                                        paths[via_depots[:, dec_pos], 0],
                                                        paths[via_depots[:, dec_pos], dec_pos]]
    assert torch.count_nonzero(sub_problem.current_capacities < 0) == 0
    lenghts += adj_matrices[torch.arange(bs), paths[:, -2], paths[:, -1]]

    return paths, via_depots, lenghts


def beam_search_decoding_loop(node_coords: Tensor, adj_matrices: Tensor, demands: Tensor, capacities: Tensor,
                              net: Module, beam_size: int, knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape  # (including repetition of begin=end node)
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=node_coords.device)
    via_depots = torch.full((bs * beam_size, num_nodes), False, dtype=torch.bool, device=node_coords.device)
    paths[:, -1] = num_nodes - 1

    initial_capacities = copy.deepcopy(capacities).unsqueeze(-1)
    probabilities = torch.zeros((bs, 1), device=node_coords.device)
    lenghts = torch.zeros(bs * beam_size, device=node_coords.device)

    sub_problem = DecodingSubPb(node_coords, adj_matrices, demands, initial_capacities, original_idxs)

    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, batch_in_prev_input, capacities, probabilities, sub_problem =\
            beam_search_decoding_step(capacities, sub_problem, net, probabilities, bs, beam_size, knns)

        paths = paths[batch_in_prev_input]
        via_depots = via_depots[batch_in_prev_input]
        adj_matrices = adj_matrices[batch_in_prev_input]
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot
        lenghts = lenghts[batch_in_prev_input]

        # compute lenghts for direct edges
        lenghts[~via_depots[:, dec_pos]] += adj_matrices[~via_depots[:, dec_pos],
                                                         paths[~via_depots[:, dec_pos], dec_pos - 1],
                                                         paths[~via_depots[:, dec_pos], dec_pos]]
        # compute lenghts for edges via depot
        lenghts[via_depots[:, dec_pos]] += adj_matrices[via_depots[:, dec_pos],
                                                        paths[via_depots[:, dec_pos], dec_pos - 1],
                                                        paths[via_depots[:, dec_pos], 0]] + \
                                           adj_matrices[via_depots[:, dec_pos],
                                                        paths[via_depots[:, dec_pos], 0],
                                                        paths[via_depots[:, dec_pos], dec_pos]]

    lenghts += adj_matrices[torch.arange(bs * beam_size), paths[:, -2], paths[:, -1]]

    lenghts = lenghts.reshape(bs, -1)
    paths = paths.reshape(bs, -1, num_nodes)
    via_depots = via_depots.reshape(bs, -1, num_nodes)
    min_lenghts = torch.argmin(lenghts, dim=1)
    return paths[torch.arange(bs), min_lenghts], via_depots[torch.arange(bs), min_lenghts],\
           lenghts[torch.arange(bs), min_lenghts]


def greedy_decoding_step(capacities: Tensor, sub_problem: DecodingSubPb, net: Module,
                         knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(capacities, sub_problem, net, knns)
    selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)

    new_subproblem, via_depot = reformat_subproblem_for_next_step(capacities, sub_problem, idx_selected, via_depot, knns)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subproblem


def prepare_input_and_forward_pass(capacities: Tensor, sub_problem: DecodingSubPb, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, node_dim = sub_problem.node_coords.shape
    if 0 < knns < num_nodes:
        bs, num_nodes, node_dim = sub_problem.node_coords.shape

        knn_indices = torch.topk(sub_problem.distance_matrices[:, :-1, 0], k=knns - 1, dim=-1, largest=False).indices
        # and add it manually
        knn_indices = torch.cat(
            [knn_indices, torch.full([bs, 1], num_nodes - 1, device=sub_problem.node_coords.device)],
            dim=-1)
        knn_coords = torch.gather(sub_problem.node_coords, 1, knn_indices.unsqueeze(dim=-1).repeat(1, 1, node_dim))
        knn_demands = torch.gather(sub_problem.demands, 1, knn_indices)
        current_capacities = sub_problem.current_capacities[:, -1]

        inputs = torch.cat([knn_coords, (knn_demands / capacities.unsqueeze(-1)).unsqueeze(-1),
                            (current_capacities.unsqueeze(-1) / capacities.unsqueeze(-1)).repeat(1, knns).unsqueeze(-1)
                            ], dim=-1)

        knn_scores = net(inputs, demands=knn_demands, remaining_capacities=current_capacities)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full((sub_problem.node_coords.shape[0], 2 * sub_problem.node_coords.shape[1]),
                            -np.inf, device=knn_coords.device)
        double_knn_indices = torch.zeros([knn_indices.shape[0], 2 * knn_indices.shape[1]], device=knn_indices.device,
                                         dtype=torch.int64)
        double_knn_indices[:, 0::2] = 2 * knn_indices
        double_knn_indices[:, 1::2] = 2 * knn_indices + 1

        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, double_knn_indices, knn_scores)

    else:
        current_capacities = sub_problem.current_capacities[:, -1]
        inputs = torch.cat([sub_problem.node_coords,
                            (sub_problem.demands / capacities.unsqueeze(-1)).unsqueeze(-1),
                            (current_capacities / capacities).unsqueeze(-1).repeat(1,
                                num_nodes).unsqueeze(-1)], dim=-1)
        scores = net(inputs, demands=sub_problem.demands, remaining_capacities=current_capacities)
    return scores


def beam_search_decoding_step(capacities: Tensor, sub_problem: DecodingSubPb, net: Module, prev_probabilities: Tensor,
                              test_batch_size: int, beam_size: int, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(capacities, sub_problem, net, knns)
    num_nodes = sub_problem.node_coords.shape[1]
    num_instances = sub_problem.node_coords.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    # repeat 2*num_nodes -> for each node we have two scores - direct edge and via depot
    probabilities = (prev_probabilities.repeat(1, 2 * num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, 2 * num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.node_coords = sub_problem.node_coords[batch_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.demands = sub_problem.demands[batch_in_prev_input]
    sub_problem.current_capacities = sub_problem.current_capacities[batch_in_prev_input]
    sub_problem.distance_matrices = sub_problem.distance_matrices[batch_in_prev_input]
    capacities = capacities[batch_in_prev_input]

    selected_nodes = torch.remainder(topk_indexes, 2 * num_nodes).unsqueeze(dim=1)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)

    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    new_subproblem, via_depot = reformat_subproblem_for_next_step(capacities, sub_problem, idx_selected, via_depot,
                                                                  knns)

    return idx_selected_original.squeeze(1), via_depot.squeeze(1), batch_in_prev_input, capacities,\
        topk_values.unsqueeze(dim=1), new_subproblem


def reformat_subproblem_for_next_step(capacities: Tensor, sub_problem: DecodingSubPb, idx_selected: Tensor,
                                      via_depot: Tensor, knns: int) -> DecodingSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    subpb_size = sub_problem.node_coords.shape[1]
    bs = sub_problem.node_coords.shape[0]
    is_selected = torch.arange(subpb_size, device=sub_problem.node_coords.device).unsqueeze(dim=0).repeat(bs, 1) ==\
                  idx_selected.repeat(1, subpb_size)

    # next begin node = just-selected node
    next_begin_node_coord = sub_problem.node_coords[is_selected].unsqueeze(dim=1)
    next_begin_demand = sub_problem.demands[is_selected].unsqueeze(dim=1)
    next_begin_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)

    # remaining nodes = the rest, minus current first node
    next_remaining_node_coords = sub_problem.node_coords[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    next_remaining_demands = sub_problem.demands[~is_selected].reshape((bs, -1))[:, 1:]
    next_remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]

    # concatenate
    next_node_coords = torch.cat([next_begin_node_coord, next_remaining_node_coords], dim=1)
    next_demands = torch.cat([next_begin_demand, next_remaining_demands], dim=1)
    next_original_idxs = torch.cat([next_begin_original_idx, next_remaining_original_idxs], dim=1)

    # update current capacities
    current_capacities = sub_problem.current_capacities[:, -1].unsqueeze(dim=1) - next_begin_demand

    # recompute capacities
    current_capacities[via_depot.bool()] = capacities.unsqueeze(-1)[via_depot.bool()] -\
                                           next_begin_demand[via_depot.bool()]
    if torch.count_nonzero(current_capacities < 0) > 0:
        print("stp")

    next_current_capacities = torch.cat([sub_problem.current_capacities, current_capacities], dim=-1)
    if knns != -1:
        num_nodes = sub_problem.distance_matrices.shape[1]

        # select row (=column) of adj matrix for just-selected node
        next_row_column = sub_problem.distance_matrices[is_selected]
        # remove distance to the selected node (=0)
        next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]

        # remove rows and columns of selected nodes
        next_adj_matrices = sub_problem.distance_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
        next_adj_matrices = next_adj_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes-1, -1)[:, 1:, :]

        # add new row on the top and remove second (must be done like this, because on dimenstons of the matrix)
        next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_adj_matrices], dim=1)

        # and add it to the beginning-
        next_row_column = torch.cat([torch.zeros(idx_selected.shape, device=next_row_column.device), next_row_column],
                                    dim=1)
        next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_adj_matrices], dim=2)
    else:
        next_adj_matrices = sub_problem.distance_matrices

    new_subproblem = DecodingSubPb(next_node_coords, next_adj_matrices, next_demands, next_current_capacities,
                                   next_original_idxs)

    return new_subproblem, via_depot