#!/bin/bash

seeds=(1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='ett_forecasts' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="ett_learned" ctransform_solver.max_iter=3 D=12 C=24 categorical=False seed=$seed
done