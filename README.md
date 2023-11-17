# BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization
We provide the code to learn to solve four standard combinatorial optimization problems: 
* the (Euclidian and Asymetric) Traveling Salesman Problems (TSPs)
* the Capacitated Vehicle Routing Problem (CVRP)
* the Orienteering Problem (OP)
* the Knapsack Problem (KP)

For BQ-Perceiver code, please check [bq-perceiver](https://github.com/naver/bq-nco/tree/perceiver) branch.

## Paper
See [BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization](https://arxiv.org/abs/2301.03313) for the paper associated with this codebase. If you find this code useful, please cite our paper as: 

 ``` 
@inproceedings{
    drakulic2023bqnco,
    title={BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization},
    author={Darko Drakulic and Sofia Michel and Florian Mai and Arnaud Sors and Jean-Marc Andreoli},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
    url={https://arxiv.org/abs/2301.03313},
}
``` 

## Quickstart

For data preparation, check [data](./data/) directory.

### Training
Using (near-) optimal trajectories
```
python train_[tsp,cvrp,op,kp].py
  --train_dataset TRAIN_DATASET
  --val_datasets VAL_DATASET
   --test_datasets TEST_DATASET
   --output_dir OUTPUT_DIR
```

### Test
To test our pretrained models:

```
python test_[tsp,cvrp,op,kp].py
  --path_to_model_to_test ./pretrained_models/[tsp,cvrp,op,kp].best
  --test_datasets TEST_DATASET
  --output_dir OUTPUT_DIR
  --test_only
```
