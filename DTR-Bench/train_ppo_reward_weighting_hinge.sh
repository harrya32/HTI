#!/bin/bash

# Script to train PPO agent with varying lambda_nk values from 0 to 10

echo "Starting reward_weighting.py runs for lambda_nk = 1,3,5"

for lambda in 1 3 5
do
    echo "Running training with lambda_nk = $lambda"
    python reward_weighting_hinge.py --lambda_nk $lambda
    echo "Completed training for lambda_nk = $lambda"
    echo "--------------------------------------------"
done

echo "All runs completed."