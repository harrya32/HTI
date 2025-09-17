#!/bin/bash

seeds=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=False wandb_project="quantile_learned" ctransform_solver.max_iter=10 D=3 C=12 categorical=False seed=$seed
done