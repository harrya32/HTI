#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py -m dataset='conditional_semicircles' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=0.4 lambda=0.05 wandb_project="semicircles_learned_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed

  python NLOT/train.py -m dataset='conditional_semicircles' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="semicircles_eucl" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed

  python NLOT/train.py -m dataset='conditional_semicircles' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=0.4 lambda=0.05 wandb_project="semicircles_eucl_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed

  python NLOT/train.py -m dataset='conditional_semicircles' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="semicircles_learned" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed

  python NLOT/train.py -m dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.4 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
done