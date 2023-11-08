"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import time
import argparse
import numpy
import numpy as np
from ortools.algorithms import pywrapknapsack_solver


def generate_instance(size, MAX_VALUE):
    values = np.random.randint(1, MAX_VALUE-1, size)
    weights = np.random.randint(1, MAX_VALUE-1, size)
    return list(values), list(weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate and solve KP dataset")
    parser.add_argument("--num_instances", type=int, default=128, help="Number of instances")
    parser.add_argument("--problem_size", type=int, default=200, help="Problem size")
    parser.add_argument("--capacity", type=int, default=25, help="Capacity")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--scale", type=int, default=1e3, help="scale")
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        help="Reorder nodes/tours. Must be reordered in training dataset")
    args = parser.parse_args()

    np.random.seed(args.seed)

    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        'KnapsackExample')
    solver.set_time_limit(100)
    capacity = args.capacity * args.scale
    start_time = time.time()
    all_data = list()

    all_capacities, all_weights, all_values, all_rewards = list(), list(), list(), list()
    all_solution_lengths = list()

    for instance_num in range(args.num_instances):
        values, weights = generate_instance(args.problem_size, args.scale)
        solver.Init(values, [weights], [capacity])
        computed_value = solver.Solve()

        packed_items = []
        packed_weights = []
        total_price, total_weight = 0., 0.
        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[i])
                total_weight += weights[i]
        total_weight = total_weight
        price = computed_value

        if args.reorder:
            solution_length = len(packed_items)
            all_capacities.append(capacity)

            sol_idx = np.full(len(weights), False)
            sol_idx[packed_items] = True

            weights = np.array(weights)[sol_idx].tolist() + np.array(weights)[~sol_idx].tolist()
            values = np.array(values)[sol_idx].tolist() + np.array(values)[~sol_idx].tolist()

            all_solution_lengths.append(solution_length)
            all_weights.append(weights)
            all_values.append(values)
        else:
            all_capacities.append(capacity)
            all_weights.append(weights)
            all_values.append(values)
            all_rewards.append(price)

    capacities = np.stack(all_capacities)
    weights = np.stack(all_weights)
    values = np.stack(all_values)

    if args.reorder:
        solution_lengths = np.stack(all_solution_lengths)
        numpy.savez_compressed(args.output_filename, scale=args.scale, capacities=capacities, weights=weights,
                               values=values, solution_lengths=solution_lengths, reorder=True)
    else:
        optimal_values = np.stack(all_rewards)
        numpy.savez_compressed(args.output_filename, scale=args.scale, capacities=capacities, weights=weights,
                               values=values, optimal_values=optimal_values, reorder=False)

    print("Done")