from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

# Command-line model names (as in the paper tables) -> configuration sections.
MODEL_KEYS = {
    "DyG-WM": "dyg_wm",
    "JODIE": "jodie",
    "DyRep": "dyrep",
    "TGAT": "tgat",
    "TGN": "tgn",
    "CAWN": "cawn",
    "EdgeBank": "edgebank",
    "TCL": "tcl",
    "GraphMixer": "graphmixer",
    "DyGFormer": "dygformer",
    "CLDG": "cldg",
    "MaskDGNN": "maskdgnn",
    "DVGMAE": "dvgmae",
    "JODIE-Bipartite": "jodie_bipartite",
}

SNAPSHOT_SSL_MODELS = ("cldg", "maskdgnn", "dvgmae")

DATASETS = (
    "wikipedia", "mooc", "lastfm", "enron", "uci", "canparl",
    "contacts", "flights", "untrade", "unvote", "uslegis",
)


def model_key(model_name: str) -> str:
    """Accept both ``DyG-WM`` and ``dyg_wm`` style names."""
    lookup = {name.lower(): key for name, key in MODEL_KEYS.items()}
    lookup.update({key: key for key in MODEL_KEYS.values()})
    try:
        return lookup[model_name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown model {model_name!r}; choose from {list(MODEL_KEYS)}"
        ) from None


def deep_update(target: dict, updates: dict) -> dict:
    """Recursively merge ``updates`` into ``target`` (mappings merge, rest replaces)."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def load_config(config_path: str | Path, dataset_name: str) -> dict:
    """Load the YAML config and apply the entry of ``dataset_name``.

    The returned mapping holds the shared protocol, every model section with
    its dataset-specific values merged in, and ``data.path``.
    """
    with Path(config_path).open() as handle:
        config = yaml.safe_load(handle) or {}
    datasets = config.pop("datasets", {}) or {}
    dataset_name = dataset_name.lower()
    if dataset_name not in datasets:
        raise ValueError(
            f"dataset {dataset_name!r} is not configured in {config_path}; "
            f"available: {sorted(datasets)}"
        )
    deep_update(config, dict(datasets[dataset_name] or {}))
    data_dir = Path(config.pop("data_dir", "processed_data"))
    config["dataset_name"] = dataset_name
    config["data"] = {"path": str(data_dir / f"{dataset_name}.npz")}
    if "window_size" in config.get("dyg_wm", {}):
        raise ValueError("set the window size under split.window_size / window_size")
    return config


def model_arguments(config: dict, key: str) -> dict:
    """Constructor keyword arguments of model ``key`` (data-dependent values
    such as ``num_nodes`` and the negative-destination pool are added by the
    training script)."""
    link = dict(config.get("link", {}))
    section = dict(config.get(key, {}))
    if key == "dyg_wm":
        return {
            **link,
            **section,
            "window_size": int(config["split"]["window_size"]),
        }
    return {**section, **link}


def training_arguments(config: dict, key: str) -> dict:
    """Shared evaluation/checkpoint settings overlaid with ``<key>_training``.

    The snapshot SSL baselines use their own complete ``<key>_training``."""
    if key in SNAPSHOT_SSL_MODELS:
        return dict(config.get(f"{key}_training", {}))
    return {
        **dict(config.get("training", {})),
        **dict(config.get(f"{key}_training", {})),
    }


# ----------------------------------------------------------------------------
# Node classification
# ----------------------------------------------------------------------------
NODE_MODEL_KEYS = {
    "DyG-WM": "dyg_wm",
    "SG-JEPA": "sg_jepa",
    "EvolveGCN-H": "evolvegcn_h",
    "ROLAND": "roland",
    "TGN": "tgn",
    "TGAT": "tgat",
    "CAWN": "cawn",
    "TCL": "tcl",
    "GraphMixer": "graphmixer",
    "DyGFormer": "dygformer",
    "CLDG": "cldg",
    "MaskDGNN": "maskdgnn",
    "DVGMAE": "dvgmae",
}

NODE_DATASETS = ("dblp", "tmall", "patent")


def node_model_key(model_name: str) -> str:
    lookup = {name.lower(): key for name, key in NODE_MODEL_KEYS.items()}
    lookup.update({key: key for key in NODE_MODEL_KEYS.values()})
    try:
        return lookup[model_name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown node model {model_name!r}; choose from {list(NODE_MODEL_KEYS)}"
        ) from None


def load_node_config(config_path: str | Path, dataset_name: str) -> dict:
    """Load the node-classification YAML and apply the entry of ``dataset_name``."""
    return load_config(config_path, dataset_name)
