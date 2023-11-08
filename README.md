# BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization

Branch for BQ-PerceiverIO model.

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
python test_[tsp,cvrp].py
  --path_to_model_to_test ./pretrained_models/[tsp,cvrp]_perceiver.best
  --test_datasets TEST_DATASET
  --output_dir OUTPUT_DIR
```
