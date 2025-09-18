#!/bin/bash

seeds=(10 11 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=False wandb_project="quantile_eucl" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
done