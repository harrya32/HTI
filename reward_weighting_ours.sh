#!/bin/bash

seeds=(3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=0.1 conditional_bandwidth=1.0 lambda=0.0001 wandb_project="reward_weighting_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed
done