#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.5 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.4 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.3 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.2 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.1 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.05 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 seed=$seed

done