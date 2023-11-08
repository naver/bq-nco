"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import numpy as np
import json
SCALE = 1e7


def create_op_file(filename, coords, prices, instance, opts):
    with open(filename, "w") as file:
        file.write("NAME: " + "seed_" + str(opts.seed) + "_instance_" + str(instance) + "\n")
        file.write("TYPE: OP\n")
        file.write("COMMENT: \n")
        file.write("DIMENSION: " + str(opts.num_nodes) + "\n")
        file.write("COST_LIMIT: " + str(SCALE * opts.tour_max_len) + "\n")
        file.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for i, node in enumerate(coords):
            file.write(str(i+1) + " " + str(int(SCALE*node[0])) + " " + str(int(SCALE*node[1])) + "\n")
        file.write("NODE_SCORE_SECTION\n")
        file.write(str(1) + " 0\n")
        for i, price in enumerate(prices):
            file.write(str(i + 2) + " " + str(price) + "\n")
        file.write("DEPOT_SECTION\n")
        file.write("1\n")
        file.write("-1\n")
        file.write("EOF\n")


def read_solution_file(filename):
    with open(filename) as file:
        solution = json.load(file)
    _score = solution["sol"]["val"]
    _tour = solution["sol"]["cycle"]
    _score = np.array(_score)
    _tour = np.array(_tour) - 1

    return _tour, _score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate and solve OP")
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--tour_max_len", type=float, default=4.)
    parser.add_argument("--working_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--price_type", type=str, default="dist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--op_exec", type=str, required=True)
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        help="Reorder nodes/tours. Must be reordered in training dataset")
    args = parser.parse_args()

    np.random.seed(args.seed)

    all_node_coords, all_upper_bounds, all_node_values, all_solution_lens, all_collected_rewards = \
        list(), list(), list(), list(), list()

    for instance in range(args.num_instances):
        node_coords = np.random.random([1 + args.num_nodes, 2])

        if args.price_type == "unif":
            prices = np.random.randint(1, 100, args.num_nodes)
        elif args.price_type == "dist":
            norm = np.linalg.norm(node_coords[0] - node_coords[1:], axis=-1)
            prices = 1 + (norm / norm.max() * 99).astype(np.int_)
        else:
            raise "Not implemented"

        op_file = os.path.join(args.working_dir, str(instance) + ".op")
        create_op_file(op_file, node_coords, prices, instance, args)
        solution_file = os.path.join(os.getcwd(), "prob.sol")
        if os.path.exists(solution_file):
            os.unlink(solution_file)
        op_cmd = os.path.join(args.op_exec) + ' opt ' + op_file
        os.system(op_cmd)

        if os.path.exists(solution_file):
            tour, score = read_solution_file(solution_file)
            os.unlink(solution_file)

            if args.reorder:
                node_coords = np.array(node_coords)
                solution_len = len(tour)
                all_node_idx = set([i for i in range(args.num_nodes + 1)])
                gt_nodes_idx = set(tour)
                remaining_nodes = all_node_idx.difference(gt_nodes_idx)
                tour = np.append(tour, list(remaining_nodes))
                tour = np.append(tour, 0)
                assert len(tour) == args.num_nodes + 2
                # reorder nodes and prices
                node_coords = node_coords[tour]
                node_values = [0] + prices.tolist()
                node_values = np.array(node_values)[tour]
            else:
                node_coords = node_coords.tolist()
                node_values = [0] + prices.tolist()

                node_coords.append(node_coords[0])
                node_values.append(0)

                node_values = np.array(node_values)
                node_coords = np.array(node_coords)
                solution_len = -1

            all_solution_lens.append(solution_len)
            all_node_coords.append(node_coords)
            all_upper_bounds.append(args.tour_max_len)
            all_node_values.append(node_values)
            all_collected_rewards.append(score)

    all_solution_lens = np.stack(all_solution_lens)
    all_node_coords = np.stack(all_node_coords)
    all_upper_bounds = np.stack(all_upper_bounds)
    all_node_values = np.stack(all_node_values) / 100
    all_collected_rewards = np.stack(all_collected_rewards) / 100

    if args.reorder:
        np.savez_compressed(args.output_file, coords=all_node_coords, upper_bounds=all_upper_bounds,
                            values=all_node_values, solution_lengths=all_solution_lens, reorder=True)
    else:
        np.savez_compressed(args.output_file, coords=all_node_coords, upper_bounds=all_upper_bounds,
                            values=all_node_values, collected_rewards=all_collected_rewards, reorder=True)
    print(instance, "done.")
