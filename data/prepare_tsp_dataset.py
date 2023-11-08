"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os.path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from data.solvers.concorde.tsp import TSPSolver
SCALE = 1e6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate and solve TSP")
    parser.add_argument("--num_instances", type=int, default=10, help="Numbers of TSP instances")
    parser.add_argument("--num_nodes", type=int, default=100, help="Numbers of nodes")
    parser.add_argument("--output_filename", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        help="Reorder nodes/tours. training dataset MUST BE reordered")

    args = parser.parse_args()
    np.random.seed(args.seed)

    all_instance_coords = np.random.random([args.num_instances, args.num_nodes, 2])

    coords, tours, tour_lens = list(), list(), list()

    for instance_coords in all_instance_coords:
        solver = TSPSolver.from_data(instance_coords[:, 0] * SCALE, instance_coords[:, 1] * SCALE, norm="EUC_2D")
        solution = solver.solve()
        solution_closed_tour = list(solution[0]) + [0]

        if args.reorder:
            coords_reordered = instance_coords[np.array(solution_closed_tour)]
            coords.append(coords_reordered)

        else:
            instance_coords = instance_coords.tolist()
            instance_coords.append(instance_coords[0])

            # compute tour length
            adj_matrix = squareform(pdist(instance_coords, metric='euclidean'))
            tour_len = sum([adj_matrix[solution_closed_tour[i], solution_closed_tour[i + 1]]
                            for i in range(len(solution_closed_tour)-1)])
            tour_lens.append(tour_len)
            coords.append(instance_coords)

    if args.reorder:
        np.savez_compressed(os.path.join(args.output_filename), coords=np.array(coords), reorder=True)
    else:
        np.savez_compressed(os.path.join(args.output_filename), coords=np.array(coords), tour_lens=tour_lens,
                            reorder=False)

    print("Data transformed and saved to " + args.output_filename)
