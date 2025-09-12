#!/bin/bash

seeds=(4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reacher_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=0.75 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False seed=$seed
done