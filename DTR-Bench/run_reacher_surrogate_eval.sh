#!/bin/bash

# Number of iterations
ITERATIONS=1

# Run name
RUN_NAME_5="nlot_metric"
RUN_NAME_4="eucl_no_potential"
RUN_NAME_3="eucl_w_potential_075"
RUN_NAME_2="learned_no_potential"
RUN_NAME_1="learned_w_potential_075"

# Python script to execute
SCRIPT="surrogate_eval_reacher.py"

# Loop to run the script 5 times
for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_1..."
    python3 $SCRIPT --name $RUN_NAME_1 --iter $i --all_lambdas
done

for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_2..."
    python3 $SCRIPT --name $RUN_NAME_2 --iter $i --all_lambdas
done

for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_3..."
    python3 $SCRIPT --name $RUN_NAME_3 --iter $i --all_lambdas
done

for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_4..."
    python3 $SCRIPT --name $RUN_NAME_4 --iter $i --all_lambdas
done

for i in $(seq 1 $ITERATIONS)
do
    echo "Running iteration $i with run name: $RUN_NAME_5..."
    python3 $SCRIPT --name $RUN_NAME_5 --iter $i --all_lambdas
done

echo "All $ITERATIONS iterations completed."