#!/usr/bin/env bash
# ============================================================
# Batch evaluation of multiple flow-model checkpoints using
# eval_heldout.py.  Edit the CHECKPOINTS array and the shared
# arguments below, then run:
#   bash run_eval_heldout_batch.sh
# ============================================================

# ---------- shared eval arguments ----------
DATA_PATH="mfm/dataloaders/diffusion_2moons_dropout.pt"
TRAIN_INDICES="0 5 10"
EVAL_INDICES="1 2 3 4 6 7 8 9"
DIM=2
HIDDEN_DIMS="64 64 64"
ACTIVATION="selu"
NUM_STEPS=200
SOLVER="euler"

# File where mean EMD for each run is appended (TSV)
RESULTS_FILE="eval_results/batch_results_cfm.tsv"

# Set to "--class_conditioning --categorical --cond_dim 1 --num_categories 2"
# or leave empty for unconditioned runs
COND_ARGS="--class_conditioning --categorical --cond_dim 1 --num_categories 2"

# ---------- checkpoints to evaluate ----------
# Each entry is an absolute or relative path to a .ckpt file.
# The out_dir for each run is derived from the run id automatically.
CHECKPOINTS=(
    "checkpoints/custom_pt/20260219_154818_diffusion_2moons_s0_r42m1qhc/flow_model/epoch=599-step=3600.ckpt"
    "checkpoints/custom_pt/20260219_154951_diffusion_2moons_s1_cktiw6bn/flow_model/epoch=439-step=2640.ckpt"
    "checkpoints/custom_pt/20260219_155113_diffusion_2moons_s2_2hcmzoj0/flow_model/epoch=709-step=4260.ckpt"
    "checkpoints/custom_pt/20260219_155300_diffusion_2moons_s3_ysh6hqmf/flow_model/epoch=499-step=3000.ckpt"
    "checkpoints/custom_pt/20260219_155428_diffusion_2moons_s4_ffj7wk0c/flow_model/epoch=1199-step=7200.ckpt"
    "checkpoints/custom_pt/20260219_155721_diffusion_2moons_s5_h1r0c37a/flow_model/epoch=569-step=3420.ckpt"
    "checkpoints/custom_pt/20260219_155852_diffusion_2moons_s6_8far1kxk/flow_model/epoch=709-step=4260.ckpt"
    "checkpoints/custom_pt/20260219_160046_diffusion_2moons_s7_o1306g80/flow_model/epoch=779-step=4680.ckpt"
    "checkpoints/custom_pt/20260219_160246_diffusion_2moons_s8_depfj5yx/flow_model/epoch=539-step=3240.ckpt"
    "checkpoints/custom_pt/20260219_160413_diffusion_2moons_s9_7x1x7g6i/flow_model/epoch=939-step=5640.ckpt"
    "checkpoints/custom_pt/20260219_160639_diffusion_2moons_s10_70bhox3k/flow_model/epoch=799-step=4800.ckpt"
    "checkpoints/custom_pt/20260219_160836_diffusion_2moons_s11_i7am1vpr/flow_model/epoch=779-step=4680.ckpt"
    "checkpoints/custom_pt/20260219_161044_diffusion_2moons_s12_pz4wiblx/flow_model/epoch=559-step=3360.ckpt"
    "checkpoints/custom_pt/20260219_161215_diffusion_2moons_s13_2fzxdw8a/flow_model/epoch=489-step=2940.ckpt"
    "checkpoints/custom_pt/20260219_161345_diffusion_2moons_s14_r755vvcc/flow_model/epoch=1039-step=6240.ckpt"
    "checkpoints/custom_pt/20260219_161628_diffusion_2moons_s15_lk500viq/flow_model/epoch=599-step=3600.ckpt"
    "checkpoints/custom_pt/20260219_161806_diffusion_2moons_s16_kwxnqh23/flow_model/epoch=629-step=3780.ckpt"
    "checkpoints/custom_pt/20260219_161946_diffusion_2moons_s17_dmuo8u1f/flow_model/epoch=469-step=2820.ckpt"
    "checkpoints/custom_pt/20260219_162109_diffusion_2moons_s18_le49x58e/flow_model/epoch=649-step=3900.ckpt"
    "checkpoints/custom_pt/20260219_162253_diffusion_2moons_s19_qpodxm4y/flow_model/epoch=569-step=3420.ckpt"
)

CHECKPOINTS=(
    #"checkpoints/custom_pt/20260301_202318_diffusion_2moons_s0_23mk6ujg/flow_model/epoch=519-step=3120.ckpt"
    
    #"checkpoints/custom_pt/20260301_203337_diffusion_2moons_s4_smg0as17/flow_model/epoch=239-step=1440.ckpt"

    #"checkpoints/custom_pt/20260301_203045_diffusion_2moons_s1_7wt5xzfb/flow_model/epoch=109-step=660.ckpt"

    #"checkpoints/custom_pt/20260301_203129_diffusion_2moons_s2_u30saa4y/flow_model/epoch=229-step=1380.ckpt"

    #"checkpoints/custom_pt/20260301_203223_diffusion_2moons_s3_0qnqf3pu/flow_model/epoch=509-step=3060.ckpt"

    #"checkpoints/custom_pt/20260301_205221_diffusion_2moons_s5_zox65v0g/flow_model/epoch=579-step=3480.ckpt"

    "checkpoints/custom_pt/20260301_205341_diffusion_2moons_s6_jt26no11/flow_model/epoch=199-step=1200.ckpt"
    "checkpoints/custom_pt/20260301_205432_diffusion_2moons_s7_2wn0kgjf/flow_model/epoch=149-step=900.ckpt"
    "checkpoints/custom_pt/20260301_205520_diffusion_2moons_s8_zdjlrynw/flow_model/epoch=349-step=2100.ckpt"
    "checkpoints/custom_pt/20260301_205622_diffusion_2moons_s9_pzik94zs/flow_model/epoch=229-step=1380.ckpt"
)

# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$(dirname "$RESULTS_FILE")"

total=${#CHECKPOINTS[@]}
echo "Running eval on $total checkpoint(s)..."
echo ""

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    # Derive a readable output directory from the checkpoint path:
    # e.g.  checkpoints/custom_pt/<run_id>/flow_model/<ckpt>.ckpt
    #   ->  eval_results/<run_id>
    RUN_ID=$(echo "$CKPT" | sed 's|.*/\([^/]*\)/flow_model/.*|\1|')
    OUT_DIR="eval_results/${RUN_ID}"

    echo "[$((i+1))/$total] Checkpoint : $CKPT"
    echo "             Output dir : $OUT_DIR"
    echo ""

    conda run -n mfm python eval_heldout.py \
        --ckpt          "$CKPT" \
        --data_path     "$DATA_PATH" \
        --train_indices $TRAIN_INDICES \
        --eval_indices  $EVAL_INDICES \
        --dim           "$DIM" \
        --hidden_dims   $HIDDEN_DIMS \
        --activation    "$ACTIVATION" \
        --num_steps     "$NUM_STEPS" \
        --solver        "$SOLVER" \
        --out_dir       "$OUT_DIR" \
        --results_file  "$RESULTS_FILE" \
        $COND_ARGS

    echo ""
    echo "------------------------------------------------------------"
    echo ""
done

echo "All done.  Aggregated results: $RESULTS_FILE"
