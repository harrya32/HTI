#!/bin/bash

seeds=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reacher_data' geometry='neural_net_metric' include_inverse_potential=False wandb_project="reacher_pooladian" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] plotting.disable=True seed=$seed
done