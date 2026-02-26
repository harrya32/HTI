#!/bin/bash

"""seeds=(4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reward_weighting_eucl" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed

  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed

  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False seed=$seed
done

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=False wandb_project="reward_weighting_hinge_learned_no_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  
  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_hinge_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=False wandb_project="reward_weighting_hinge_eucl_no_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='reward_weighting_hinge_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_hinge_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
done"""

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reacher_data' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reacher_eucl_no_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='reacher_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reacher_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py dataset='reacher_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='reacher_data' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="reacher_learned_no_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

  python NLOT/train.py dataset='reacher_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reacher_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed
  python NLOT/train.py dataset='reacher_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=2.0 conditional_bandwidth=1.0 lambda=0.001 wandb_project="reacher_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=11 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] seed=$seed

done