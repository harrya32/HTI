#!/bin/bash

for seed in 5 6 7 8 9; do
    echo "========================================="
    echo "Running with seed=${seed}"
    echo "========================================="
    python -m mfm.train.main \
        --config_path ./configs/custom/diffusion_2moons.yaml \
        --seeds ${seed}
done
