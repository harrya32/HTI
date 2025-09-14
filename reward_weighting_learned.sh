#!/bin/bash

seeds=(0 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="reward_weighting_learned" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed
done