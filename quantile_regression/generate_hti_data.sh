#!/bin/bash
QUANTILES="0.01 0.1 0.25 0.5 0.75 0.9 0.99"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAIN_SCRIPT_PATH="$SCRIPT_DIR/generate_hti_data.py"

echo "Starting the data generation for all models..."
echo "======================================================"

if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "Error: Script not found at $TRAIN_SCRIPT_PATH"
    exit 1
fi

# Loop through each quantile in the list
for q in $QUANTILES
do
  echo ""
  echo "------------------------------------------------------"
  echo ">>> Generating data for quantile: $q"
  echo "------------------------------------------------------"
  
  python "$TRAIN_SCRIPT_PATH" --quantile $q
  
  if [ $? -ne 0 ]; then
    echo "Error: Generation failed for quantile $q. Aborting."
    exit 1
  fi
done

echo ""
echo "======================================================"
echo "All data generated."
echo "======================================================"