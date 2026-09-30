#!/usr/bin/env bash
# Train and evaluate one model for node classification on DBLP / Tmall / Patent
# at the labelled-node ratios 0.4 / 0.6 / 0.8.
#
#   bash run_node_classification.sh                    # DyG-WM, all datasets, GPU 0
#   MODEL=SG-JEPA GPU=1 bash run_node_classification.sh dblp
#   RATIOS="0.4" bash run_node_classification.sh tmall
#
# Environment variables:
#   MODEL   DyG-WM | SG-JEPA | EvolveGCN-H | ROLAND | TGN | TGAT | CAWN | TCL |
#           GraphMixer | DyGFormer | CLDG | MaskDGNN | DVGMAE
#   GPU     GPU index, -1 for CPU (default 0)
#   RATIOS  labelled-node training fractions (default "0.4 0.6 0.8")
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-DyG-WM}"
GPU="${GPU:-0}"
RATIOS="${RATIOS:-0.4 0.6 0.8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(dblp tmall patent)
fi

for dataset in "${DATASETS[@]}"; do
  for ratio in $RATIOS; do
    "$PYTHON_BIN" train_node_classification.py \
      --dataset_name "$dataset" \
      --model_name "$MODEL" \
      --train_ratio "$ratio" \
      --gpu "$GPU"
  done
done
