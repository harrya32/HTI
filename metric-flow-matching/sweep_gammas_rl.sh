#!/bin/bash

for gamma in 0.001 0.01 0.125 0.25 0.5 1 3; do
    echo "========================================="
    echo "Running with gamma=${gamma}"
    echo "========================================="
    python -m mfm.train.main \
        --config_path ./configs/custom/reacher.yaml \
        --gammas ${gamma}

    python -m mfm.train.main \
        --config_path ./configs/custom/reward_weighting.yaml \
        --gammas ${gamma}

    python -m mfm.train.main \
        --config_path ./configs/custom/reward_weighting_hinge.yaml \
        --gammas ${gamma}
done
