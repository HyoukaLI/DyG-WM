#!/usr/bin/env bash
# Rebuild every archive in processed_data/ from the raw files in DG_data/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p processed_data

for name in wikipedia mooc lastfm; do
  "$PYTHON_BIN" preprocess_data/preprocess_bipartite.py \
    --input "DG_data/${name}/${name}.csv" \
    --output "processed_data/${name}.npz"
done

for specification in \
  "CanParl:canparl" "Contacts:contacts" "Flights:flights" "UNtrade:untrade" \
  "UNvote:unvote" "USLegis:uslegis" "enron:enron" "uci:uci"
do
  raw_name="${specification%%:*}"
  output_name="${specification##*:}"
  "$PYTHON_BIN" preprocess_data/preprocess_homogeneous.py \
    --dataset-dir "DG_data/${raw_name}" \
    --output "processed_data/${output_name}.npz"
done
