#!/bin/bash

# Number of iterations
ITERATIONS=1

# Run name
RUN_NAME_1="2_5_eucl_w_potential"

# Python script to execute
SCRIPT="surrogate_eval_reacher_more_anchors.py"

# Loop to run the script 5 times
for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_1..."
    python3 $SCRIPT --name $RUN_NAME_1 --iter $i --all_lambdas
done

echo "All $ITERATIONS iterations completed."