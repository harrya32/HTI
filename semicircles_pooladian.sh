#!/bin/bash

seeds=(16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=False wandb_project="semicircles_pooladian" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
done