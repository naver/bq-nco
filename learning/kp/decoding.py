"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module

@dataclass
class DecodingSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    values: Tensor
    weights: Tensor
    remaining_capacities: Tensor
    original_idxs: Tensor


def decode(values: Tensor, weights: Tensor, capacities: Tensor, scale: Tensor, net: Module,
           beam_size: int = 1) -> Tensor:

    if beam_size == 1:
        selected_items, prices = greedy_decoding(values, weights, capacities, scale, net)
    else:
        selected_items, prices = beam_search_decoding(values, weights, capacities, net, beam_size)

    return selected_items, prices


def greedy_decoding(values: Tensor, weights: Tensor, capacities: Tensor, scale: Tensor, net: Module):
    bs, problem_size = values.shape
    original_idxs = torch.tensor(list(range(problem_size)), device=values.device)[None, :].repeat(bs, 1)

    sub_problem = DecodingSubPb(values, weights, capacities, original_idxs)
    selected_items = torch.full((bs, problem_size), -1, dtype=torch.long, device=values.device)
    steps = list()
    for dec_pos in range(problem_size):
        scores, idx_selected, sub_problem = greedy_decoding_step(sub_problem, scale, net)
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break
        selected_items[:, dec_pos] = idx_selected
        steps.append([sub_problem.original_idxs, scores])

    # trick: add 0 at the end of weights and values, for "selected nodes" with index -1
    selected_items[selected_items == -1] = problem_size
    weights = torch.cat([weights, torch.zeros(bs, 1, device=weights.device)], dim=-1)
    values = torch.cat([values, torch.zeros(bs, 1, device=values.device)], dim=-1)
    assert torch.all(torch.gather(weights, 1, selected_items).sum(axis=1) <= capacities)

    rewards = torch.gather(values, 1, selected_items).sum(axis=1)

    return selected_items, rewards


def greedy_decoding_step(sub_problem: DecodingSubPb, scale: Tensor, net: Module) -> (Tensor, DecodingSubPb):
    values = sub_problem.values / scale.unsqueeze(-1)
    weights = sub_problem.weights / scale.unsqueeze(-1)
    current_capacities = sub_problem.remaining_capacities / scale

    inputs = torch.cat([values.unsqueeze(-1), weights.unsqueeze(-1),
                        current_capacities.unsqueeze(dim=-1).repeat(1, weights.shape[1]).unsqueeze(-1)],
                       dim=-1)
    scores = net(inputs, weights=sub_problem.weights, remaining_capacities=sub_problem.remaining_capacities)
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)
    # if all possible choices are masked (= -inf), than instance is solved
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return scores, idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected)


def beam_search_decoding(values: Tensor, weights: Tensor, capacities: Tensor, net: Module, beam_size: int) -> Tensor:
    bs, problem_size = values.shape
    assert bs == 1 # this should be improved

    original_idxs = torch.tensor(list(range(problem_size)), device=values.device)[None, :].repeat(bs, 1)

    sub_problem = DecodingSubPb(values, weights, capacities, original_idxs)
    trajectories = torch.full((bs * beam_size, problem_size), -1, dtype=torch.long, device=values.device)
    probabilities = torch.zeros((bs, 1), device=values.device)

    # select first elements for all batch instances
    rewards = torch.zeros(bs * beam_size, device=values.device)
    possible_trajectories, possible_rewards, possible_capacities = list(), list(), list()
    for dec_pos in range(problem_size):
        idx_selected, selected_values, idx_done, pos_in_previous_input, probabilities, sub_problem =\
            beam_search_decoding_step(sub_problem, net, probabilities, beam_size, trajectories[:, :dec_pos])
        if idx_selected is None:
            break
            # all tours are done!

        trajectories = trajectories[pos_in_previous_input]
        rewards = rewards[pos_in_previous_input]
        trajectories[:, dec_pos] = idx_selected

        rewards += selected_values
        possible_trajectories.extend(trajectories)
        possible_rewards.extend(rewards)
        possible_capacities.extend(sub_problem.capacities)

    possible_rewards = torch.stack(possible_rewards)
    possible_trajectories = torch.stack(possible_trajectories)
    possible_capacities = torch.tensor(possible_capacities)

    possible_trajectories = possible_trajectories[possible_capacities > 0]
    possible_rewards = possible_rewards[possible_capacities > 0]

    return possible_trajectories[torch.argmax(possible_rewards)], torch.max(possible_rewards)


def beam_search_decoding_step(sub_problem: DecodingSubPb, net: Module, prev_probabilities: Tensor,
                              beam_size: int, trajectories: Tensor) -> (Tensor, DecodingSubPb):
    num_remaining_items = sub_problem.values.shape[1]
    inputs = torch.cat([sub_problem.values.unsqueeze(-1), sub_problem.weights.unsqueeze(-1),
                        sub_problem.capacities.unsqueeze(dim=-1).repeat(1, sub_problem.weights.shape[1]).unsqueeze(-1)],
                       dim=-1)
    scores = net(inputs)

    idx_done = (scores.max(dim=-1)[0] == -np.inf).nonzero()
    if len(idx_done) == len(scores):
        return None, None, idx_done, None, None, None

    candidates = torch.softmax(scores, dim=-1)

    probabilities = (prev_probabilities.repeat(1, num_remaining_items) +
                     torch.log(candidates)).reshape(sub_problem.values.shape[0], -1)
    # set all probabilities in finished trajectories to -inf

    top_probabilities, top_indexes = torch.sort(probabilities.reshape(1, -1), descending=True)

    top_indexes = top_indexes[(~top_probabilities.isnan() & ~top_probabilities.isinf())]
    top_probabilities = top_probabilities[(~top_probabilities.isnan() & ~top_probabilities.isinf())]

    top_probabilities = top_probabilities.flatten()
    top_indexes = top_indexes.flatten()
    pos_in_prev_input = torch.div(top_indexes, num_remaining_items, rounding_mode="floor")
    idx_selected = torch.remainder(top_indexes, num_remaining_items)
    sub_problem.values = sub_problem.values[pos_in_prev_input]
    sub_problem.weights = sub_problem.weights[pos_in_prev_input]
    sub_problem.capacities = sub_problem.capacities[pos_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[pos_in_prev_input]

    selected_values = sub_problem.values[torch.arange(sub_problem.values.shape[0]), idx_selected.squeeze(-1)]
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected.unsqueeze(-1)).squeeze(-1)

    # select non-intersected trajectories
    trajectories = trajectories[pos_in_prev_input]
    trajectories = torch.cat([trajectories, idx_selected_original.unsqueeze(-1)], dim=-1)

    unique_trajectories, selected_idx = list(), list()
    no_selected = 0
    for idx in range(len(trajectories)):
        trajectory = trajectories[idx].cpu().tolist()
        trajectory.sort()
        if trajectory not in unique_trajectories:
            unique_trajectories.append(trajectory)
            selected_idx.append(idx)
            no_selected += 1
        if no_selected == beam_size:
            break

    idx_selected_original = idx_selected_original[selected_idx]
    idx_selected = idx_selected[selected_idx].unsqueeze(-1)
    selected_values = selected_values[selected_idx]
    pos_in_prev_input = pos_in_prev_input[selected_idx]
    sub_problem.values = sub_problem.values[selected_idx]
    sub_problem.weights = sub_problem.weights[selected_idx]
    sub_problem.capacities = sub_problem.capacities[selected_idx]
    sub_problem.original_idxs = sub_problem.original_idxs[selected_idx]
    top_probabilities = top_probabilities[selected_idx].unsqueeze(-1)

    return idx_selected_original, selected_values, idx_done, pos_in_prev_input, top_probabilities,\
        reformat_subproblem_for_next_step(sub_problem, idx_selected)


def reformat_subproblem_for_next_step(sub_problem: DecodingSubPb, idx_selected: Tensor) -> DecodingSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    subpb_size = sub_problem.weights.shape[1]
    bs = sub_problem.weights.shape[0]
    is_selected = torch.arange(
        subpb_size, device=sub_problem.weights.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1,
                                                                                                             subpb_size)
    # remaining items = the rest
    next_capacities = sub_problem.remaining_capacities - sub_problem.weights[is_selected]
    next_remaining_values = sub_problem.values[~is_selected].reshape((bs, -1))
    next_remaining_weights = sub_problem.weights[~is_selected].reshape((bs, -1))
    next_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))
    return DecodingSubPb(next_remaining_values, next_remaining_weights, next_capacities, next_original_idxs)
