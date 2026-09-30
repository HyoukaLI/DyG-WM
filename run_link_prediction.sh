#!/usr/bin/env bash
# Train and evaluate one model on every link-prediction dataset.
#
#   bash run_link_prediction.sh                 # DyG-WM, random negatives, GPU 0
#   MODEL=DyGFormer GPU=1 bash run_link_prediction.sh
#   STRATEGY=historical bash run_link_prediction.sh wikipedia uci
#
# Environment variables:
#   MODEL     DyG-WM | JODIE | DyRep | TGAT | TGN | CAWN | EdgeBank | TCL | GraphMixer | DyGFormer
#   GPU       GPU index, -1 for CPU (default 0)
#   STRATEGY  evaluation negatives: random | historical | inductive (default random)
#   SETTING   transductive | inductive (default transductive)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-DyG-WM}"
GPU="${GPU:-0}"
STRATEGY="${STRATEGY:-random}"
SETTING="${SETTING:-transductive}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(wikipedia mooc lastfm enron uci canparl contacts flights untrade unvote uslegis)
fi

for dataset in "${DATASETS[@]}"; do
  "$PYTHON_BIN" train_link_prediction.py \
    --dataset_name "$dataset" \
    --model_name "$MODEL" \
    --negative_sample_strategy "$STRATEGY" \
    --setting "$SETTING" \
    --gpu "$GPU"
done
