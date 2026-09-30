#!/usr/bin/env bash
# Download the node-classification inputs (DBLP / Tmall / Patent).
#
#   bash preprocess_data/download_node_data.sh                # all three
#   bash preprocess_data/download_node_data.sh dblp tmall     # a subset
#
# The files are hosted on a GitHub release (REPO / RELEASE below):
#   dblp    processed archive           -> processed_data/dblp.npz (ready to use)
#   tmall   DeepWalk features (2 parts) -> DG_data/tmall/tmall.npy
#   patent  DeepWalk features           -> DG_data/patent/patent.npy
# Afterwards build the Tmall / Patent archives:
#   python preprocess_data/preprocess_spikenet_node.py --dataset tmall
#   python preprocess_data/preprocess_spikenet_node.py --dataset patent
# The Tmall edge list and labels are included in DG_data/tmall/; the Patent raw
# files (patent_edges.json, patent_nodes.json) come from the SpikeNet release.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REPO="${REPO:-HyoukaLI/DyGJEPA}"
RELEASE="${RELEASE:-node-data-v1}"
BASE="https://github.com/$REPO/releases/download/$RELEASE"

fetch() {  # <asset> <destination>
  if [[ -s "$2" ]]; then echo "  have $2"; return 0; fi
  mkdir -p "$(dirname "$2")"
  echo "  get  $BASE/$1"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$2.partial" "$BASE/$1" && mv "$2.partial" "$2"
}

DATASETS=("$@"); [[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(dblp tmall patent)
for dataset in "${DATASETS[@]}"; do
  echo "== $dataset"
  case "$dataset" in
    dblp)
      fetch dblp.npz processed_data/dblp.npz
      ;;
    tmall)
      target=DG_data/tmall/tmall.npy
      if [[ -s "$target" ]]; then
        echo "  have $target"
      else
        fetch tmall.npy.part_aa "$target.part_aa"
        fetch tmall.npy.part_ab "$target.part_ab"
        cat "$target.part_aa" "$target.part_ab" > "$target"
        rm -f "$target.part_aa" "$target.part_ab"
      fi
      ;;
    patent)
      fetch patent.npy DG_data/patent/patent.npy
      ;;
    *)
      echo "unknown dataset $dataset (expected dblp, tmall or patent)" >&2
      exit 1
      ;;
  esac
done
