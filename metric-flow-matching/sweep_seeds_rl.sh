#!/bin/bash

for seed in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    echo "========================================="
    echo "Running with seed=${seed}"
    echo "========================================="
    python -m mfm.train.main \
        --config_path ./configs/custom/reacher.yaml \
        --seeds ${seed}

    #python -m mfm.train.main \
    #    --config_path ./configs/custom/reward_weighting.yaml \
    #    --seeds ${seed}

    #python -m mfm.train.main \
    #    --config_path ./configs/custom/reward_weighting_hinge.yaml \
    #    --seeds ${seed}
done
