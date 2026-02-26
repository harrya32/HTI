#!/bin/bash

seeds=(1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reward_weighting_eucl" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed

  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed

  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed
done