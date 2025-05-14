#!/bin/bash

# Define the lambda values you want to test
lambda_values=(7.5 11 12 13 14 15)

# Run the script for each lambda value
for lambda in "${lambda_values[@]}"
do
   echo "==================================="
   echo "Running with lambda_nk = $lambda"
   echo "==================================="
   python reward_weighting.py --lambda_nk $lambda
done

echo "All runs completed!"
