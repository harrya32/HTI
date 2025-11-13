#!/bin/bash

seeds=(1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data_long' geometry='neural_net_metric_eig' num_train_iters=201 include_inverse_potential=False wandb_project="quantile_long_learned" ctransform_solver.max_iter=5 D=12 C=12 categorical=False target_potential_dim_hidden=[64,64,64,64,64,64,64,64] source_map_dim_hidden=[64,64,64,64,64,64,64,64] seed=$seed
done