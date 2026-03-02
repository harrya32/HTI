#!/bin/bash

for seed in 1; do
    echo "========================================="
    echo "Running with seed=${seed}"
    echo "========================================="
    python -m mfm.train.main \
        --config_path ./configs/custom/quantile_data.yaml \
        --seeds ${seed}
done
