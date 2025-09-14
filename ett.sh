#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='ett_forecasts' geometry='neural_net_metric_eig' num_train_iters=1000 include_inverse_potential=True bandwidth=0.75 conditional_bandwidth=1.0 lambda=0.001 wandb_project="ett_learned_w_potential" ctransform_solver.max_iter=3 D=12 C=24 categorical=False seed=$seed
  python NLOT/train.py dataset='ett_forecasts' geometry='neural_net_metric_eig' num_train_iters=1000 include_inverse_potential=False wandb_project="ett_learned" ctransform_solver.max_iter=3 D=12 C=24 categorical=False seed=$seed
done