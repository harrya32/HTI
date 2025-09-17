#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=True bandwidth=0.4 lambda=0.01 wandb_project="quantile_eucl_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=True bandwidth=0.4 conditional_bandwidth=0.1 lambda=0.01 wandb_project="quantile_eucl_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=0.1 lambda=0.01 wandb_project="quantile_eucl_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
done