#!/bin/bash

seeds=(10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=True wandb_project="2moons_learned_w_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="2moons_eucl_no_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=True wandb_project="2moons_eucl_w_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="2moons_learned_no_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=False wandb_project="2moons_pooladian" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=True wandb_project="2moons_nlot_metric" seed=$seed

done