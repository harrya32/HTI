#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="reward_weighting_learned" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done