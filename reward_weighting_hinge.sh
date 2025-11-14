#!/bin/bash

seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='sq_euclidean_manifold' num_train_iters=501 include_inverse_potential=False bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_hinge_eucl_no_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done