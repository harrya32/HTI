#!/bin/bash

seeds=(3 4 5 6 7 8 9)

for seed in "${seeds[@]}"; do
  #python NLOT/train.py num_train_iters=1001 dataset='reacher_all_data' time_points=[0.5,1] geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py num_train_iters=1001 dataset='reacher_all_data' time_points=[0.5,1] geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="reacher_learned_no_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  #python NLOT/train.py num_train_iters=1001 dataset='reacher_all_data' time_points=[0.5,1] geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py num_train_iters=1001 dataset='reacher_all_data' time_points=[0.5,1] geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reacher_eucl_no_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done