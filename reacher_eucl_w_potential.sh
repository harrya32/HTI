#!/bin/bash

seeds=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reacher_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done