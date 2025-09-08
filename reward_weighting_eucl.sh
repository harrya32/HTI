#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reward_weighting_eucl" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed
done