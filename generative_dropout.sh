#!/bin/bash

seeds=(13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_learned_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  #python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="2moons_eucl_no_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  #python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_eucl_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  #python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="2moons_learned_no_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  #python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=False wandb_project="2moons_pooladian" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

done