"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from utils.misc import compute_tour_lens

@dataclass
class DecodingSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    node_coords: Tensor
    original_idxs: Tensor
    dist_matrices: Tensor


def decode(node_coords: Tensor, dist_matrices: Tensor, net: Module, beam_size: int, knns: int) -> Tensor:
    if beam_size == 1:
        tours = greedy_decoding_loop(node_coords, dist_matrices, net, knns)
    else:
        tours = beam_search_decoding_loop(node_coords, dist_matrices, net, beam_size, knns)

    tours = tours[:, :-1]

    num_nodes = tours.shape[1]
    assert tours.sum(axis=1).sum() == tours.shape[0] * .5 * (num_nodes - 1) * num_nodes
    # compute distances by using (original) distance matrix
    distances = compute_tour_lens(tours, dist_matrices)

    return tours, distances


def beam_search_decoding_loop(node_coords: Tensor, dist_matrices: Tensor, net: Module, beam_size: int,
                              knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape  # (including repetition of begin=end node)

    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1

    probabilities = torch.zeros((bs, 1), device=node_coords.device)
    distances = torch.zeros(bs * beam_size, 1, device=node_coords.device)

    sub_problem = DecodingSubPb(node_coords, original_idxs, dist_matrices)
    for dec_pos in range(1, num_nodes - 1):
        origin_coords = sub_problem.node_coords[:, 0]

        idx_selected_original, batch_in_prev_input, probabilities, sub_problem =\
            beam_search_decoding_step(sub_problem, net, probabilities, bs, beam_size, knns)

        paths = paths[batch_in_prev_input]
        paths[:, dec_pos] = idx_selected_original
        distances = distances[batch_in_prev_input]
        # these are distances between normalized! coordinates (!= real tour lengths)
        distances += torch.cdist(origin_coords[batch_in_prev_input].unsqueeze(dim=1),
                                 sub_problem.node_coords[:, 0].unsqueeze(dim=1)).squeeze(-1)
    distances += torch.cdist(sub_problem.node_coords[:, 0].unsqueeze(dim=1),
                             sub_problem.node_coords[:, -1].unsqueeze(dim=1)).squeeze(-1)

    distances = distances.reshape(bs, -1)
    paths = paths.reshape(bs, -1, num_nodes)
    return paths[torch.arange(bs), torch.argmin(distances, dim=1)]


def greedy_decoding_loop(node_coords: Tensor, dist_matrices: Tensor, net: Module, knns: int) -> Tensor:
    num_nodes = node_coords.shape[1]  # (including repetition of begin=end node)
    bs = node_coords.shape[0]
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1

    sub_problem = DecodingSubPb(node_coords, original_idxs, dist_matrices)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        paths[:, dec_pos] = idx_selected

    return paths


def prepare_input_and_forward_pass(sub_problem: DecodingSubPb, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, node_dim = sub_problem.node_coords.shape
    if 0 < knns < num_nodes:
        # sort node by distance from the origin (ignore the target node)
        _, sorted_nodes_idx = torch.sort(sub_problem.dist_matrices[:, :-1, 0], dim=-1)

        # select KNNs
        knn_indices = sorted_nodes_idx[:, :knns-1]
        # and add the target at the end
        input_nodes = torch.cat([knn_indices, torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1)

        node_coords = torch.gather(sub_problem.node_coords, 1, input_nodes.unsqueeze(dim=-1).repeat(1, 1, node_dim))
        knn_scores = net(node_coords)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full(sub_problem.node_coords.shape[:-1], -np.inf, device=node_coords.device)
        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, knn_indices, knn_scores)
    else:
        scores = net(sub_problem.node_coords)

    return scores


def beam_search_decoding_step(sub_problem: DecodingSubPb, net: Module, prev_probabilities: Tensor, test_batch_size: int,
                              beam_size: int, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    num_nodes = sub_problem.node_coords.shape[1]
    num_instances = sub_problem.node_coords.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    probabilities = (prev_probabilities.repeat(1, num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.node_coords = sub_problem.node_coords[batch_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.dist_matrices = sub_problem.dist_matrices[batch_in_prev_input]
    idx_selected = torch.remainder(topk_indexes, num_nodes).unsqueeze(dim=1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected).squeeze(-1)

    return idx_selected_original, batch_in_prev_input, topk_values.unsqueeze(dim=1), \
           reformat_subproblem_for_next_step(sub_problem, idx_selected, knns)


def greedy_decoding_step(sub_problem: DecodingSubPb, net: Module, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)  # (b, 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected, knns)


def reformat_subproblem_for_next_step(sub_problem: DecodingSubPb, idx_selected: Tensor, knns: int) -> DecodingSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    subpb_size = sub_problem.node_coords.shape[1]
    bs = sub_problem.node_coords.shape[0]
    is_selected = torch.arange(
        subpb_size, device=sub_problem.node_coords.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1,
                                                                                                                 subpb_size)
    # next begin node = just-selected node
    next_begin_node_coord = sub_problem.node_coords[is_selected].unsqueeze(dim=1)
    next_begin_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)

    # remaining nodes = the rest, minus current first node
    next_remaining_node_coords = sub_problem.node_coords[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    next_remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]

    # concatenate
    next_node_coords = torch.cat([next_begin_node_coord, next_remaining_node_coords], dim=1)
    next_original_idxs = torch.cat([next_begin_original_idx, next_remaining_original_idxs], dim=1)

    if knns != -1:
        num_nodes = sub_problem.dist_matrices.shape[1]

        # select row (=column) of adj matrix for just-selected node
        next_row_column = sub_problem.dist_matrices[is_selected]
        # remove distance to the selected node (=0)
        next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]

        # remove rows and columns of selected nodes
        next_dist_matrices = sub_problem.dist_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
        next_dist_matrices = next_dist_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes-1, -1)[:, 1:, :]

        # add new row on the top and remove second (must be done like this, because on dimenstons of the matrix)
        next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_dist_matrices], dim=1)

        # and add it to the beginning-
        next_row_column = torch.cat([torch.zeros(idx_selected.shape, device=next_row_column.device),
                                     next_row_column], dim=1)
        next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_dist_matrices], dim=2)
    else:
        next_dist_matrices = sub_problem.dist_matrices

    return DecodingSubPb(next_node_coords, next_original_idxs, next_dist_matrices)