#!/bin/bash

seeds=(15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.1 conditional_bandwidth=1.0 lambda=0.0001 wandb_project="reward_weighting_nlot_metric" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done