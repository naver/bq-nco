# Data preparation

## Using your data 
If you want to test our model with your data, you must to prepare them in a specific format.
For routing problems data of first node must be copied to the end. E.g. in case of CVRP 100, data should be in the format:

```
coords = [[x0, y0], [x1, y1], ..., [x99, y99], [x0, y0]]
demands = [0, d1, ..., d99, 0]
capacities = c
tour_lens = tl
reorder = False 
```

If you want to train our model with your data, training data must be reordered by using solution trajectories (see prepare_*_dataset.py files)  

### TSP

We used code [PyConcorde](https://github.com/jvkersch/pyconcorde), a Python wrapper around the Concorde TSP solver for solve TSP instances. 
It must be installed in ./solvers directory.

To create training dataset: 
```
python prepare_tsp_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --output_filename ./output
       --seed 123
       --reorder
```

To create validation/test dataset: 
```
python prepare_tsp_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --output_filename ./output/file.npz
       --seed 777
```
 
### CVRP
For CVRP, we need a solver as a standalone application.
It should be installed and path to executable file provided to the script.
We used [LKH](http://webhotel4.ruc.dk/~keld/research/LKH-3/) solver.

To create training dataset:
```
python prepare_cvrp_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --capacity 50
       --working_dir ./tmp
       --output_filename ./output/file.npz
       --lkh_exec PATH_TO_LKH
       --seed 123
       --reorder
```

To create validation/test dataset: 
```
python prepare_cvrp_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --capacity 50
       --working_dir ./tmp
       --output_filename ./output/file.npz
       --lkh_exec PATH_TO_LKH
       --seed 777
```


### OP

For OP, we used [op-splver](https://github.com/gkobeaga/op-solver).
Like in case of CVRP, path to the executable file must be provided to the script.

To create training dataset:
```
python prepare_op_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --tour_max_len 4
       --working_dir ./tmp
       --output_filename ./output/file.npz
       --OP_exec PATH_TO_PATH_SOLVER
       --seed 123
       --reorder
```

To create validation/test dataset: 
```
python prepare_op_dataset.py 
       --num_instances 10000
       --num_nodes 100
       --tour_max_len 4
       --working_dir ./tmp
       --output_filename ./output/file.npz
       --OP_exec PATH_TO_PATH_SOLVER
       --seed 777
```

### KP
For KP, we use [ORTools](https://developers.google.com/optimization) solver.
To create training dataset:
```
python prepare_kp_dataset.py 
       --num_instances 10000
       --problem_size 100
       --capacity 20
       --output_filename ./output/file.npz
       --seed 123
       --reorder
```

To create validation/test dataset: 
```
python prepare_kp_dataset.py 
       --num_instances 10000
       --problem_size 100
       --capacity 20
       --output_filename ./output/file.npz
       --seed 777
```
