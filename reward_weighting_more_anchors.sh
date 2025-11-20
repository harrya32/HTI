#!/bin/bash

seeds=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py num_train_iters=2001 time_points=[0.2,0.5,0.8] dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py num_train_iters=2001 time_points=[0.2,0.5,0.8] dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=False bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_learned_no_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py num_train_iters=2001 time_points=[0.2,0.5,0.8] dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py num_train_iters=2001 time_points=[0.2,0.5,0.8] dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=False bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_eucl_no_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done