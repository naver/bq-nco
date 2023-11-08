"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass
import numpy as np
import torch
import copy
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
    node_values: Tensor
    upper_bounds: Tensor
    original_idxs: Tensor
    dist_matrices: Tensor


def decode(node_coords: Tensor, node_values: Tensor, upper_bounds: Tensor, dist_matrices: Tensor, net: Module,
           beam_size: int, knns: int) -> Tensor:
    if beam_size == 1:
        tours, collected_rewards = greedy_decoding_loop(node_coords, node_values, upper_bounds, dist_matrices, net, knns)
    else:
        tours, collected_rewards = beam_search_decoding_loop(node_coords, node_values, upper_bounds, dist_matrices, net, beam_size, knns)

    distances = compute_tour_lens(tours, dist_matrices)
    assert torch.all(distances <= upper_bounds + 1e-4)

    return collected_rewards, tours


def greedy_decoding_loop(node_coords: Tensor, node_values: Tensor, upper_bounds: Tensor, dist_matrices: Tensor,
                         net: Module, knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    collected_rewards = torch.zeros(bs, device=node_coords.device)
    sub_problem = DecodingSubPb(node_coords, node_values, upper_bounds, original_idxs, dist_matrices)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        paths[:, dec_pos] = idx_selected
        collected_rewards += node_values[torch.arange(bs), idx_selected]
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break

    return paths, collected_rewards


def greedy_decoding_step(sub_problem: DecodingSubPb, net: Module, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)  # (b, 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected)


def beam_search_decoding_loop(node_coords: Tensor, node_values: Tensor, upper_bounds: Tensor, dist_matrices: Tensor,
                              net: Module, beam_size: int, knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape  # (including repetition of begin=end node)

    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1

    probabilities = torch.zeros((bs, 1), device=node_coords.device)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    sub_problem = DecodingSubPb(node_coords, node_values, upper_bounds, original_idxs, dist_matrices)
    solutions_candidates = [None] * bs

    for dec_pos in range(1, num_nodes - 1):
        idx_selected_original, instances_done, batch_in_prev_input, probabilities, sub_problem = \
            beam_search_decoding_step(sub_problem, net, probabilities, bs, beam_size, knns)

        for idx in torch.nonzero(instances_done):
            instance_idx = torch.div(idx, beam_size, rounding_mode="trunc")
            if solutions_candidates[instance_idx.item()] is None:
                solutions_candidates[instance_idx.item()] = list()
            solutions_candidates[instance_idx.item()].append(copy.deepcopy(paths[idx]))
        if torch.count_nonzero(instances_done) == bs * beam_size:
            # all done
            break

        paths = paths[batch_in_prev_input]
        paths[:, dec_pos] = idx_selected_original

    paths, collected_rewards = list(), list()
    for instance_id in range(bs):
        tours_s = torch.cat(solutions_candidates[instance_id], dim=0)
        rewards = node_values[instance_id][tours_s].sum(axis=-1)
        best = torch.argmax(rewards)
        paths.append(tours_s[best])
        collected_rewards.append(rewards[best])

    return torch.stack(paths, dim=0), torch.stack(collected_rewards, dim=0)


def beam_search_decoding_step(sub_problem: DecodingSubPb, net: Module, prev_probabilities: Tensor, test_batch_size: int,
                              beam_size: int, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    instances_done = scores.max(dim=-1)[0] == -np.inf
    num_nodes = sub_problem.node_coords.shape[1]
    num_instances = sub_problem.node_coords.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    probabilities = (prev_probabilities.repeat(1, num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)
    # replace nans (scores of already done instances) to -np.inf
    probabilities[probabilities.isnan()] = -np.inf

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.node_coords = sub_problem.node_coords[batch_in_prev_input]
    sub_problem.node_values = sub_problem.node_values[batch_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.dist_matrices = sub_problem.dist_matrices[batch_in_prev_input]
    sub_problem.upper_bounds = sub_problem.upper_bounds[batch_in_prev_input]
    idx_selected = torch.remainder(topk_indexes, num_nodes).unsqueeze(dim=1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected).squeeze(-1)

    idx_selected_original[topk_values == -np.inf] = 0
    return idx_selected_original, instances_done, batch_in_prev_input, topk_values.unsqueeze(dim=1),\
        reformat_subproblem_for_next_step(sub_problem, idx_selected)


def prepare_input_and_forward_pass(sub_problem: DecodingSubPb, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, node_dim = sub_problem.node_coords.shape
    if 0 < knns < num_nodes:
        # sort node by distance from the origin (ignore the target node)
        _, sorted_nodes_idx = torch.sort(sub_problem.dist_matrices[:, :-1, 0], dim=-1)

        num_closest = (knns - 1) // 2
        num_largest = knns - 1 - num_closest
        # select KNNs
        indices_closest = sorted_nodes_idx[:, :num_closest]
        remaining_idx = sorted_nodes_idx[:, num_closest:-1]
        indices_largest = torch.stack([remaining_idx[i][torch.topk(sub_problem.node_values[i, remaining_idx[i]],
                           k=num_largest)[1]] for i in range(bs)])

        # concatenate them, together with the destination
        input_nodes_idx = torch.cat([indices_closest, indices_largest,
                                     torch.full([bs, 1], num_nodes - 1, device=indices_closest.device)], dim=-1)
        knn_node_coords = torch.gather(sub_problem.node_coords, 1, input_nodes_idx.unsqueeze(dim=-1).repeat(1, 1,
                                                                                                            node_dim))
        knn_node_values = torch.gather(sub_problem.node_values, 1, input_nodes_idx)
        knn_dist_matrices = torch.gather(sub_problem.dist_matrices, 1,
                                         input_nodes_idx.unsqueeze(-1).repeat(1, 1, num_nodes))
        knn_dist_matrices = torch.gather(knn_dist_matrices, 2, input_nodes_idx.unsqueeze(1).repeat(1, knns, 1))
        inputs = torch.cat([knn_node_coords, knn_node_values.unsqueeze(-1),
                            sub_problem.upper_bounds.unsqueeze(-1).repeat(1, knns).unsqueeze(-1)], dim=-1)
        knn_scores = net(inputs, dist_matrices=knn_dist_matrices)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full(sub_problem.node_coords.shape[:-1], -np.inf, device=knn_node_coords.device)
        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, input_nodes_idx, knn_scores)
    else:
        inputs = torch.cat([sub_problem.node_coords, sub_problem.node_values.unsqueeze(-1),
                            sub_problem.upper_bounds.unsqueeze(-1).repeat(1,
                                                                          sub_problem.node_coords.shape[1]).unsqueeze(
                                -1)],
                           dim=-1)
        scores = net(inputs, dist_matrices=sub_problem.dist_matrices)

    return scores


def reformat_subproblem_for_next_step(sub_problem: DecodingSubPb, idx_selected: Tensor) -> DecodingSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _ = sub_problem.node_coords.shape
    is_selected = torch.arange(
        subpb_size, device=sub_problem.node_coords.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1,
                                                                                                                 subpb_size)
    # next begin node = just-selected node
    next_begin_node_coord = sub_problem.node_coords[is_selected].unsqueeze(dim=1)
    next_begin_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)
    next_begin_node_value = sub_problem.node_values[is_selected].unsqueeze(dim=1)

    # remaining nodes = the rest, minus current first node
    next_remaining_node_coords = sub_problem.node_coords[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    next_remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]
    next_remaining_node_values = sub_problem.node_values[~is_selected].reshape((bs, -1))[:, 1:]

    # concatenate
    next_node_coords = torch.cat([next_begin_node_coord, next_remaining_node_coords], dim=1)
    next_original_idxs = torch.cat([next_begin_original_idx, next_remaining_original_idxs], dim=1)
    next_node_values = torch.cat([next_begin_node_value, next_remaining_node_values], dim=1)

    next_upper_bounds = sub_problem.upper_bounds - sub_problem.dist_matrices[:, 0, :][is_selected]

    # select row (=column) of adj matrix for just-selected node
    next_row_column = sub_problem.dist_matrices[is_selected]
    # remove distance to the selected node (=0)
    next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]

    # remove rows and columns of selected nodes
    next_dist_matrices = sub_problem.dist_matrices[~is_selected].reshape(bs, -1, subpb_size)[:, 1:, :]
    next_dist_matrices = next_dist_matrices.transpose(1, 2)[~is_selected].reshape(bs, subpb_size-1, -1)[:, 1:, :]

    # add new row on the top and remove second (must be done like this, because on dimenstons of the matrix)
    next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_dist_matrices], dim=1)

    # and add it to the beginning-
    next_row_column = torch.cat([torch.zeros(idx_selected.shape, device=next_row_column.device), next_row_column], dim=1)
    next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_dist_matrices], dim=2)

    return DecodingSubPb(next_node_coords, next_node_values, next_upper_bounds, next_original_idxs, next_dist_matrices)