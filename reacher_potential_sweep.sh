#!/bin/bash

bandwidths=(0.1 0.2 0.5 1)

for bandwidth in "${bandwidths[@]}"; do
  python NLOT/train.py dataset='reacher_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=$bandwidth conditional_bandwidth=1.0 lambda=0.0001 wandb_project="reacher_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False
done