#!/bin/bash

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='neural_net_metric' num_train_iters=1001 include_inverse_potential=False wandb_project="reward_weighting_hinge_pooladian" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done