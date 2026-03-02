#!/usr/bin/env bash

# ---------- shared eval arguments ----------
DATA_PATH="mfm/dataloaders/quantile_data_new.pt"


# ---------- checkpoints to evaluate ----------
# Each entry is an absolute or relative path to a .ckpt file.
# The out_dir for each run is derived from the run id automatically.
CHECKPOINTS=(
    "checkpoints/custom_pt/20260224_110017_quantile_data_new_s0_ywsfm3a8/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_124856_quantile_data_new_s1_i6zgh0sw/flow_model/epoch=1989-step=15920.ckpt"
    "checkpoints/custom_pt/20260224_110713_quantile_data_new_s2_1cu78hmc/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_111054_quantile_data_new_s3_av0ak9zu/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_111430_quantile_data_new_s4_ys847qjo/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_111802_quantile_data_new_s5_mfa38u8o/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_112126_quantile_data_new_s6_1iocndsv/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_112523_quantile_data_new_s7_epauui7c/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_112917_quantile_data_new_s8_iqt5fsw8/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_113245_quantile_data_new_s9_slwq89xh/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_123912_quantile_data_new_s10_q9dtyfmx/flow_model/epoch=2749-step=22000.ckpt"
    "checkpoints/custom_pt/20260224_114049_quantile_data_new_s11_p26g2q36/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_114451_quantile_data_new_s12_0582qcd2/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_114829_quantile_data_new_s13_8qdsuse3/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_115153_quantile_data_new_s14_ly4tq7ee/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_115523_quantile_data_new_s15_4aizsk8d/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_115856_quantile_data_new_s16_rqhzxj1m/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_120222_quantile_data_new_s17_3yjo14zp/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_120614_quantile_data_new_s18_fhkrqzb9/flow_model/epoch=1999-step=16000.ckpt"
    "checkpoints/custom_pt/20260224_121034_quantile_data_new_s19_55qck2da/flow_model/epoch=1999-step=16000.ckpt"
)

# ---------- shared results file ----------
RESULTS_FILE="eval_results/quantile_results.tsv"

# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

total=${#CHECKPOINTS[@]}
echo "Running eval on $total checkpoint(s)..."
echo "Results will be saved to: $RESULTS_FILE"
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

    conda run -n mfm python eval_quantile.py \
        --ckpt          "$CKPT" \
        --data_path     "$DATA_PATH" \
        --out_dir       "$OUT_DIR" \
        --results_file  "$RESULTS_FILE"
    echo ""
    echo "------------------------------------------------------------"
    echo ""
done

echo ""
echo "============================================================"
echo "Aggregated results across all checkpoints ($RESULTS_FILE):"
echo "============================================================"
conda run -n mfm python - <<'PYEOF'
import sys, os, numpy as np

results_file = os.environ.get("RESULTS_FILE", "eval_results/quantile_results.tsv")
# allow the variable to be passed via env; fall back to the hard-coded path
import argparse
p = argparse.ArgumentParser()
p.add_argument("--results_file", default=results_file)
# parse only the known flag if someone passes it; otherwise use default
args, _ = p.parse_known_args()
rf = args.results_file

if not os.path.exists(rf):
    print(f"Results file not found: {rf}")
    sys.exit(1)

with open(rf) as f:
    lines = [l.rstrip("\n") for l in f if l.strip()]

header = lines[0].split("\t")
rows = [l.split("\t") for l in lines[1:]]

# Identify numeric columns (everything except ckpt / out_dir)
numeric_cols = [h for h in header if h not in ("ckpt", "out_dir")]
col_idx = {h: header.index(h) for h in numeric_cols}

print(f"{'Column':<20} {'Mean':>12} {'Std':>12} {'N':>5}")
print("-" * 52)
for col in numeric_cols:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[col_idx[col]]))
        except (ValueError, IndexError):
            pass
    if vals:
        print(f"{col:<20} {np.mean(vals):>12.6f} {np.std(vals):>12.6f} {len(vals):>5}")
PYEOF