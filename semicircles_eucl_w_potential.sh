#!/bin/bash

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='conditional_semicircles' geometry='sq_euclidean_manifold' plotting.disable=False num_train_iters=5001 include_inverse_potential=True bandwidth=0.05 lambda=0.05 wandb_project="semicircles_eucl_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
done