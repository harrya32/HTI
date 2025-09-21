#!/bin/bash

seeds=(15)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="quantile_learned_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
done