<h1 align="center">DyG-WM</h1>

<h3 align="center">
  Predictive Dynamic Graph World Modelling: From Temporal Context to Future Relations
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg" />
  <img src="https://img.shields.io/badge/Task-Dynamic%20Link%20Prediction-orange.svg" />
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

The baselines are taken from [DyGLib](https://github.com/yule-BUAA/DyGLib) and
need no extra packages.

---

## 📊 Benchmarks & Preprocessing

DyG-WM is evaluated on **11 real-world dynamic graphs**:

- **Bipartite**: Wikipedia, MOOC, LastFM
- **Homogeneous**: Enron, UCI, Can. Parl., Contact, Flights, UN Trade, UN Vote, US Legis.

The processed archives are **included** in `processed_data/`, so the
experiments run right after cloning. To rebuild them from the raw data, see
[`DG_data/DATASETS_README.md`](DG_data/DATASETS_README.md) and run

```bash
bash preprocess_data/preprocess_all_data.sh
```

Each event stream is cut into 50 snapshots of equal event count and split
chronologically 70 / 15 / 15. Duplicate events and exact timestamps are kept,
so the continuous-time baselines still consume the original event stream.

---

## 🚀 Running the Code

The code supports

- dynamic link prediction in the **transductive** and **inductive** (new-node) settings,
- **random / historical / inductive** negative sampling at evaluation
  ([Poursafaei et al., NeurIPS'22](https://openreview.net/forum?id=1GVpwr2Tfdg)),
- DyG-WM and the baselines JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL,
  GraphMixer and DyGFormer under one protocol.

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

### Running all datasets

```bash
bash run_link_prediction.sh                                   # DyG-WM on all 11 datasets
MODEL=DyGFormer GPU=1 bash run_link_prediction.sh             # a baseline
STRATEGY=historical bash run_link_prediction.sh wikipedia uci # a subset
```

### Useful options

| Option | Meaning |
| --- | --- |
| `--model_name` | `DyG-WM`, `JODIE`, `DyRep`, `TGAT`, `TGN`, `CAWN`, `EdgeBank`, `TCL`, `GraphMixer`, `DyGFormer` |
| `--gpu` | GPU index, `-1` for CPU, `auto` for CUDA → MPS → CPU |
| `--seeds 0 1 2` | override the default seeds `0 1 2 3 4` |
| `--num_epochs` | override the number of training epochs |
| `--config` | configuration file (default `configs/link_prediction.yaml`) |

All hyperparameters, including the dataset-specific settings of DyG-WM and the
DyGLib-reported settings of the baselines, are in
[`configs/link_prediction.yaml`](configs/link_prediction.yaml).

---

## 📁 Repository Structure

```text
├── configs/link_prediction.yaml   # protocol, model and dataset-specific hyperparameters
├── models/
│   ├── DyGWM.py                   # DyG-WM
│   ├── dygwm_layers.py            # snapshot encoder, RWPE / time encodings, path signature
│   ├── TGAT.py                    # TGAT
│   ├── EdgeBank.py                # EdgeBank
│   ├── DyGLibAdapter.py           # event-stream wrapper for the DyGLib backbones
│   └── CAWN.py DyGFormer.py GraphMixer.py MemoryModel.py TCL.py modules.py   # DyGLib (MIT)
├── utils/
│   ├── DataLoader.py              # snapshot data structures and .npz loader
│   ├── link_utils.py              # windows, split, query sampling, structural statistics, metrics
│   ├── negative_sampling.py       # DyGLib historical / inductive negatives
│   ├── inductive_setting.py       # DyGLib new-node setting
│   ├── temporal_utils.py          # event streams and temporal neighbour index
│   ├── neighbor_sampler.py        # DyGLib neighbour sampler
│   ├── load_configs.py
│   └── utils.py
├── preprocess_data/               # raw data -> processed_data/*.npz
├── processed_data/                # processed archives (included)
├── DG_data/                       # raw data (only needed for preprocessing)
├── train_link_prediction.py
├── evaluate_link_prediction.py
└── run_link_prediction.sh
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

---

## 🤝 Acknowledgements

The baselines and the evaluation protocol build on
[DyGLib](https://github.com/yule-BUAA/DyGLib) (MIT license, see
`models/LICENSE_DyGLib`). We thank the authors for making their code
publicly available.
