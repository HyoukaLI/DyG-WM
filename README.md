<h1 align="center">DyG-WM</h1>

<h3 align="center">
  Predictive Dynamic Graph World Modelling: From Temporal Context to Future Relations
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg" />
  <img src="https://img.shields.io/badge/Task-Dynamic%20Graphs-orange.svg" />
</p>

---

## 🔍 Overview

This repository contains the official implementation of **DyG-WM**.

TL;DR: DyG-WM casts dynamic link prediction as **predictive world modelling**
in latent space. From a window of past graph snapshots, the model predicts the
latent state of the *future* snapshot with a JEPA-style objective, where time
itself plays the role of the mask: the online branch only sees the context
snapshots, and stop-gradient EMA encoders of the next snapshot provide the
targets. Two design choices make this useful for link prediction:

- **Relation-centric targets.** The predicted unit is the future *relation*
  of a node pair, not only its endpoints. A pair-conditioned subgraph pools
  the context around both endpoints, and a directed relation encoder maps the
  endpoint states to a relation latent that the predictor must forecast.
- **Order- and multiplicity-aware paths.** Depth-2 path signatures summarise
  the ordered structural path of every node and every candidate pair across
  the context, and a causal event history (counts, recency and event features
  strictly before each query timestamp) restores the multiplicity and
  direction that de-duplicated snapshots discard.

The link head scores a candidate `(u, v, t)` with a survival-style intensity
built on the relation latent, the causal history and the forecasted context.
For node classification the same world model is trained with the node-level
JEPA objective and read out with a frozen probe.

---

## ⚙️ Requirements

Tested with Python 3.10 and PyTorch ≥ 2.1 (CUDA or CPU; Apple MPS also works).

```bash
conda create -n dygwm python=3.10 -y
conda activate dygwm
# install the PyTorch build that matches your CUDA version first, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The baselines need no extra packages. Regenerating the DeepWalk node
features for node classification additionally needs
`pip install gensim numba scipy tqdm` (not needed to run the experiments).

---

## 📊 Benchmarks & Preprocessing

**Dynamic link prediction** — 11 real-world dynamic graphs:

- **Bipartite**: Wikipedia, MOOC, LastFM
- **Homogeneous**: Enron, UCI, Can. Parl., Contact, Flights, UN Trade, UN Vote, US Legis.

These processed archives are **included** in `processed_data/`, so the link
experiments run right after cloning. Each event stream is cut into 50
snapshots of equal event count and split chronologically 70 / 15 / 15;
duplicate events and exact timestamps are kept, so the continuous-time
baselines still consume the original event stream. To rebuild them from the
raw data run `bash preprocess_data/preprocess_all_data.sh`.

**Dynamic node classification** — DBLP, Tmall and Patent under the
SpikeNet / SG-JEPA protocol (cumulative snapshots with 80-d DeepWalk node
features). These archives are large and are prepared with

```bash
bash preprocess_data/download_node_data.sh
python preprocess_data/preprocess_spikenet_node.py --dataset tmall
python preprocess_data/preprocess_spikenet_node.py --dataset patent
```

See [`DG_data/DATASETS_README.md`](DG_data/DATASETS_README.md) for sources and details.

---

## 🚀 Running the Code

The code supports

- dynamic link prediction in the **transductive** and **inductive** (new-node) settings,
- **random / historical / inductive** negative sampling at evaluation
  ([Poursafaei et al., NeurIPS'22](https://openreview.net/forum?id=1GVpwr2Tfdg)),
- dynamic node classification at labelled-node ratios 0.4 / 0.6 / 0.8,
- DyG-WM and all baselines under one protocol per task:

| Task | Baselines (`--model_name`) |
| --- | --- |
| Link prediction | `JODIE`, `DyRep`, `TGAT`, `TGN`, `CAWN`, `EdgeBank`, `TCL`, `GraphMixer`, `DyGFormer`, `CLDG`, `MaskDGNN`, `DVGMAE`, `JODIE-Bipartite` (original bipartite JODIE; Wikipedia / MOOC / LastFM only) |
| Node classification | `SG-JEPA`, `EvolveGCN-H`, `ROLAND`, `TGN`, `TGAT`, `CAWN`, `TCL`, `GraphMixer`, `DyGFormer`, `CLDG`, `MaskDGNN`, `DVGMAE` |

JODIE, DyRep, TGN, CAWN, TCL, GraphMixer and DyGFormer are the
[DyGLib](https://github.com/yule-BUAA/DyGLib) implementations. MaskDGNN and
DVGMAE are paper-level reimplementations (no public code was available); their
result files record this.

Evaluation follows DyGLib: one negative per positive event, the negative
destination drawn from all destinations of the stream, AP / AUC averaged over
batches of 200 positive events, and seeds 0–4.

### 🔗 Dynamic Link Prediction (Training)

```bash
python train_link_prediction.py \
  --dataset_name ${dataset_name} \
  --model_name DyG-WM \
  --gpu ${gpu}
```

For every seed this trains with random negatives, keeps the checkpoint with
the best validation AP/AUC (saved to `saved_models/`), and writes the test
metrics to `saved_results/<model>/<dataset>/` together with the mean ± std
over seeds. Logs go to `logs/`.

To report a different evaluation protocol directly after training, add

```bash
  --negative_sample_strategy historical   # or inductive
  --setting inductive                     # DyGLib new-node setting (trains on the reduced graph)
```

### 📈 Dynamic Link Prediction (Evaluation)

Re-score the saved checkpoints under any negative sampling strategy without
retraining:

```bash
python evaluate_link_prediction.py \
  --dataset_name ${dataset_name} \
  --model_name DyG-WM \
  --negative_sample_strategy ${negative_sample_strategy} \
  --gpu ${gpu}
```

Checkpoints of the inductive setting are evaluated with `--setting inductive`.

### Running all link datasets

```bash
bash run_link_prediction.sh                                   # DyG-WM on all 11 datasets
MODEL=DyGFormer GPU=1 bash run_link_prediction.sh             # a baseline
STRATEGY=historical bash run_link_prediction.sh wikipedia uci # a subset
```

### 🧠 Dynamic Node Classification (Training)

```bash
python train_node_classification.py \
  --dataset_name ${dataset_name} \
  --model_name DyG-WM \
  --train_ratio 0.4 \
  --gpu ${gpu}
```

`--dataset_name` is `dblp`, `tmall` or `patent`; `--train_ratio` is the
labelled-node training fraction (0.4 / 0.6 / 0.8). Self-supervised models are
read out on the final snapshot with a frozen MLP probe; the checkpoint is
chosen on validation macro-F1 and the final probe is fitted on
train+validation. EvolveGCN-H and ROLAND are trained with the labels (epoch
chosen on validation, then refit on train+validation). Default seeds are
42, 44, 46, 48, 50.

### 🧪 Dynamic Node Classification (Evaluation)

```bash
python evaluate_node_classification.py \
  --dataset_name ${dataset_name} \
  --model_name DyG-WM \
  --train_ratio 0.4 \
  --gpu ${gpu}
```

All datasets and ratios for one model:

```bash
bash run_node_classification.sh                        # DyG-WM
MODEL=SG-JEPA bash run_node_classification.sh dblp     # a baseline on one dataset
```

### Useful options

| Option | Meaning |
| --- | --- |
| `--model_name` | `DyG-WM`, `JODIE`, `DyRep`, `TGAT`, `TGN`, `CAWN`, `EdgeBank`, `TCL`, `GraphMixer`, `DyGFormer` |
| `--gpu` | GPU index, `-1` for CPU, `auto` for CUDA → MPS → CPU |
| `--seeds 0 1 2` | override the default seeds|
| `--num_epochs` | override the number of training epochs |
| `--config` | configuration file |

All hyperparameters, including the dataset-specific settings of DyG-WM and the
DyGLib-reported settings of the baselines, are in
[`configs/link_prediction.yaml`](configs/link_prediction.yaml) and
[`configs/node_classification.yaml`](configs/node_classification.yaml).

---

## 📁 Repository Structure

```text
├── configs/
│   ├── link_prediction.yaml       # link protocol, model and dataset-specific hyperparameters
│   └── node_classification.yaml   # node protocol and hyperparameters
├── models/
│   ├── DyGWM.py                   # DyG-WM
│   ├── layers.py                  # GraphSAGE encoder, RWPE / time encodings, path signature, PLIF
│   ├── SGJEPA.py                  # SG-JEPA
│   ├── TGAT.py                    # TGAT (link prediction)
│   ├── JODIE.py                   # original bipartite JODIE
│   ├── EdgeBank.py                # EdgeBank
│   ├── SnapshotSSL.py             # CLDG, MaskDGNN, DVGMAE
│   ├── NodeBaselines.py           # EvolveGCN-H, ROLAND and the node-classification TGN/TGAT/DyGLib wrappers
│   ├── DyGLibAdapter.py           # event-stream wrapper for the DyGLib backbones
│   └── CAWN.py DyGFormer.py GraphMixer.py MemoryModel.py TCL.py modules.py   # DyGLib (MIT)
├── utils/
│   ├── DataLoader.py              # snapshot data structures and .npz loader
│   ├── link_utils.py              # windows, split, query sampling, structural statistics, metrics
│   ├── negative_sampling.py       # DyGLib historical / inductive negatives
│   ├── inductive_setting.py       # DyGLib new-node setting
│   ├── temporal_utils.py          # event streams and temporal neighbour index
│   ├── neighbor_sampler.py        # DyGLib neighbour sampler
│   ├── node_evaluation.py         # stratified split, frozen MLP probes, F1
│   ├── load_configs.py
│   └── utils.py
├── preprocess_data/               # raw data -> processed_data/*.npz (link and node)
├── processed_data/                # processed link archives (included); node archives go here too
├── DG_data/                       # raw data (only needed for preprocessing)
├── train_link_prediction.py   evaluate_link_prediction.py   run_link_prediction.sh
└── train_node_classification.py   evaluate_node_classification.py   run_node_classification.sh
```

---

## 📌 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{dygwm,
  title   = {Predictive Dynamic Graph World Modelling: From Temporal Context to Future Relations},
  author  = {},
  journal = {},
  year    = {}
}
```
