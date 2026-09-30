"""Train DyG-WM or a baseline for dynamic node classification.

Example:
    python train_node_classification.py --dataset_name dblp --model_name DyG-WM \
        --train_ratio 0.4 --gpu 0

Protocol (SpikeNet / SG-JEPA): self-supervised models are trained on the
snapshot sequence and read out on the final snapshot with a frozen MLP probe;
the labelled nodes are split into train / validation / test (``train_ratio``
labelled fraction), the checkpoint is chosen on validation macro-F1, and the
final probe is fitted on train+validation and scored on test.  EvolveGCN-H and
ROLAND are trained end-to-end with the labels (epoch chosen on validation,
then refit on train+validation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.DyGWM import DyGWM
from models.NodeBaselines import (
    DyGLibStatelessNodeSSL,
    EvolveGCNHNodeClassifier,
    ROLANDNodeClassifier,
    TGATNodeSSL,
    TGNNodeSSL,
)
from models.SGJEPA import SGJEPA
from models.SnapshotSSL import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
    SnapshotSSLLinkBaseline,
)
from utils.DataLoader import DynamicGraph, load_npz
from utils.load_configs import (
    NODE_DATASETS,
    NODE_MODEL_KEYS,
    load_node_config,
    node_model_key,
)
from utils.node_evaluation import (
    ProbeSplit,
    final_probe,
    final_probe_ensemble,
    macro_micro_f1,
    require_aligned_views,
    stratified_split,
    validation_probe,
    validation_probe_ensemble,
)
from utils.utils import (
    cpu_state_dict,
    create_logger,
    device_description,
    get_device,
    release_device_memory,
    set_random_seed,
)

JEPA_MODELS = ("dyg_wm", "sg_jepa")
SUPERVISED_MODELS = ("evolvegcn_h", "roland")
TEMPORAL_SSL_MODELS = ("tgn", "tgat", "cawn", "tcl", "graphmixer", "dygformer")
SNAPSHOT_SSL_MODELS = ("cldg", "maskdgnn", "dvgmae")


# ----------------------------------------------------------------------------
# Readout helpers
# ----------------------------------------------------------------------------
def _require_global_node_order(node_ids: torch.Tensor, num_nodes: int) -> None:
    expected = torch.arange(num_nodes, device=node_ids.device)
    if not torch.equal(node_ids, expected):
        raise ValueError(
            "node classification requires every node to be active in the final target snapshot"
        )


def _infer_views(model: nn.Module, graph) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if isinstance(model, SGJEPA):
        return model.infer_node_views(graph)
    return model.infer_node_views(graph.snapshots[-model.window_size :])


def _select_node_embeddings(
    model: nn.Module,
    graph,
    probe_split: ProbeSplit,
    probe_epochs: int,
    seed: int,
    probe_hidden_dim,
    candidates: tuple[str, ...],
    ensemble: bool = False,
) -> tuple[torch.Tensor, dict[str, float], str]:
    """Score the checkpoint candidates with validation labels only.

    With ``ensemble=True`` (DyG-WM) the criterion is the validation macro-F1
    of the logit ensemble over ``candidates`` -- the same functional that is
    reported at test time.  Otherwise the best single view is kept.
    """
    if ensemble and len(candidates) < 2:
        raise ValueError("ensemble checkpoint selection needs at least two views")
    with torch.no_grad():
        views, node_ids = _infer_views(model, graph)
    _require_global_node_order(node_ids, graph.num_nodes)
    missing = [name for name in candidates if name not in views]
    if missing:
        raise KeyError(f"model is missing checkpoint views: {missing}")

    if ensemble:
        selected_views = {name: views[name] for name in candidates}
        validation = validation_probe_ensemble(
            selected_views, graph.labels, probe_split, probe_epochs, seed, probe_hidden_dim
        )
        label = f"logit_ensemble[{','.join(candidates)}]"
        selected = dict(validation)
        selected["selected_view"] = label
        return views[candidates[0]].detach(), selected, label

    best_name = candidates[0]
    best_embeddings = views[best_name]
    best_validation = validation_probe(
        best_embeddings, graph.labels, probe_split, probe_epochs, seed, probe_hidden_dim
    )
    for name in candidates[1:]:
        embeddings = views[name]
        validation = validation_probe(
            embeddings, graph.labels, probe_split, probe_epochs, seed, probe_hidden_dim
        )
        if validation["macro_f1"] > best_validation["macro_f1"]:
            best_name = name
            best_embeddings = embeddings
            best_validation = validation
    selected = dict(best_validation)
    selected["selected_view"] = best_name
    return best_embeddings.detach(), selected, best_name


def _checkpoint_views(model: nn.Module, training: dict) -> tuple[str, ...]:
    if isinstance(model, DyGWM):
        views = tuple(training.get("dyg_wm_checkpoint_views", ("prediction", "encoder")))
        require_aligned_views(views, tuple(training.get("dyg_wm_readout_views", ())))
        return views
    return ("encoder", "prediction")


def _classification_metrics(logits, labels, indices) -> dict[str, float]:
    prediction = logits[indices].argmax(dim=-1)
    macro, micro = macro_micro_f1(labels[indices], prediction, int(labels.max().item()) + 1)
    return {"macro_f1": macro, "micro_f1": micro}


def _temporal_embeddings(model: nn.Module, seed: int) -> torch.Tensor:
    if isinstance(model, (TGATNodeSSL, DyGLibStatelessNodeSSL)):
        return model.node_embeddings(seed)
    return model.node_embeddings()


# ----------------------------------------------------------------------------
# Final (test) evaluation of a selected / refitted model
# ----------------------------------------------------------------------------
def final_node_evaluation(
    key: str,
    model: nn.Module,
    graph,
    probe_split: ProbeSplit,
    training: dict,
    settings: dict,
    seed: int,
    best_view: str | None,
) -> dict:
    """Test metrics of a model whose selected checkpoint is already loaded."""
    model.eval()
    if key == "dyg_wm":
        names = tuple(training.get("dyg_wm_readout_views", ()))
        if len(names) < 2:
            raise ValueError("training.dyg_wm_readout_views needs at least two views")
        with torch.no_grad():
            views, node_ids = _infer_views(model, graph)
        _require_global_node_order(node_ids, graph.num_nodes)
        missing = [name for name in names if name not in views]
        if missing:
            raise KeyError(f"DyG-WM readout is missing views: {missing}")
        result = final_probe_ensemble(
            {name: views[name] for name in names},
            graph.labels,
            probe_split,
            int(training.get("probe_epochs", 100)),
            seed,
            training.get("probe_hidden_dim"),
        )
        result["selected_view"] = f"logit_ensemble[{','.join(names)}]"
        result["protocol"] = "ssl_multiscale_probe"
        result["supervision"] = "self_supervised_latent"
        return result
    if key == "sg_jepa":
        with torch.no_grad():
            views, node_ids = _infer_views(model, graph)
        _require_global_node_order(node_ids, graph.num_nodes)
        result = final_probe(
            views[best_view],
            graph.labels,
            probe_split,
            int(training.get("probe_epochs", 100)),
            seed,
            training.get("probe_hidden_dim"),
        )
        result["selected_view"] = best_view
        result["protocol"] = "ssl_probe"
        result["supervision"] = "self_supervised_latent"
        return result
    if key in SUPERVISED_MODELS:
        with torch.no_grad():
            test_logits = model(graph.snapshots)
        result = _classification_metrics(test_logits, graph.labels, probe_split.test)
        result.update(
            {
                "protocol": "supervised_node_classification",
                "supervision": "node_labels",
                "architecture": (
                    "evolvegcn_h_topk_matrix_gru"
                    if key == "evolvegcn_h"
                    else "roland_graphsage_hierarchical_gru"
                ),
            }
        )
        return result
    probe_hidden_dim = training.get("probe_hidden_dim")
    probe_epochs = int(training.get("probe_epochs", 100))
    if key in TEMPORAL_SSL_MODELS:
        embeddings = _temporal_embeddings(model, seed)
        result = final_probe(
            embeddings, graph.labels, probe_split, probe_epochs, seed, probe_hidden_dim
        )
        result.update(
            {
                "selected_view": "temporal_embedding",
                "protocol": "ssl_link_pretrain_then_node_probe",
                "supervision": "self_supervised_edges",
                "snapshot_adapter": "undirected_new_edges_with_tied_snapshot_time",
            }
        )
        if isinstance(model, DyGLibStatelessNodeSSL):
            result["implementation"] = "vendored_dyglib_official_backbone"
            result["node_readout"] = "self_conditioned_final_time"
        return result
    with torch.no_grad():
        embeddings = model.encode_context(graph.snapshots)
    result = final_probe(
        embeddings, graph.labels, probe_split, probe_epochs, seed, probe_hidden_dim
    )
    result.update(
        {
            "selected_view": "final_snapshot_embedding",
            "protocol": "snapshot_ssl_then_node_probe",
            "supervision": "self_supervised_snapshots",
            "implementation": model.implementation,
        }
    )
    return result


# ----------------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------------
def train_jepa(
    name: str,
    model: nn.Module,
    graph,
    windows,
    probe_split: ProbeSplit,
    training: dict,
    seed: int,
    logger,
) -> tuple[dict[str, torch.Tensor], int, str]:
    """DyG-WM / SG-JEPA: self-supervised training with probe-based checkpointing."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    is_dygwm = isinstance(model, DyGWM)
    checkpoint_views = _checkpoint_views(model, training)
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    best_view = checkpoint_views[0]
    min_checkpoint_epoch = int(training.get("min_checkpoint_epoch", 1))
    eval_every = int(training.get("eval_every", 1))
    dense_eval_epochs = int(training.get("dense_eval_epochs", 0))
    probe_epochs = int(training.get("selection_probe_epochs", 50))
    batch_size = training.get("node_batch_size")
    probe_hidden_dim = training.get("probe_hidden_dim")

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, SGJEPA):
            loss, metrics = model.loss(graph, batch_size=batch_size)
        else:
            loss, metrics = model.node_loss_windows(windows, node_batch_size=batch_size)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        if hasattr(model, "update_target_encoder"):
            model.update_target_encoder()

        should_evaluate = (
            epoch == 1
            or epoch <= dense_eval_epochs
            or epoch % eval_every == 0
            or epoch == int(training["epochs"])
        )
        if not should_evaluate:
            continue
        model.eval()
        _, validation, view_name = _select_node_embeddings(
            model,
            graph,
            probe_split,
            probe_epochs,
            seed,
            probe_hidden_dim,
            checkpoint_views,
            ensemble=is_dygwm,
        )
        logger.info(
            json.dumps(
                {"model": name, "epoch": epoch, "train": metrics, "validation_probe": validation}
            )
        )
        if validation["macro_f1"] > best_score and epoch >= min_checkpoint_epoch:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_view = view_name

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    return best_state, best_epoch, best_view


def train_supervised(
    name: str,
    model: nn.Module,
    graph,
    probe_split: ProbeSplit,
    settings: dict,
    logger,
) -> tuple[dict[str, torch.Tensor], int, dict]:
    """EvolveGCN-H / ROLAND: select the epoch on validation, then reset to the
    initialisation and retrain for exactly that many epochs on train+validation."""
    initial_state = cpu_state_dict(model)
    initial_cpu_rng = torch.random.get_rng_state()
    initial_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def optimizer_for(module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            module.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings.get("weight_decay", 0.0)),
        )

    optimizer = optimizer_for(model)
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    best_score, best_epoch, best_validation = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.snapshots)
        loss = torch.nn.functional.cross_entropy(
            logits[probe_split.train], graph.labels[probe_split.train]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                validation_logits = model(graph.snapshots)
                validation = _classification_metrics(
                    validation_logits, graph.labels, probe_split.validation
                )
            logger.info(
                json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train": {"loss": float(loss.detach())},
                        "validation": validation,
                    }
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_validation = validation
    if best_epoch == 0 or best_validation is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")

    model.load_state_dict(initial_state)
    torch.random.set_rng_state(initial_cpu_rng)
    if initial_cuda_rng is not None:
        torch.cuda.set_rng_state_all(initial_cuda_rng)
    optimizer = optimizer_for(model)
    full_train = torch.cat([probe_split.train, probe_split.validation])
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.snapshots)
        loss = torch.nn.functional.cross_entropy(logits[full_train], graph.labels[full_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    extra = {
        "validation_macro_f1": best_validation["macro_f1"],
        "validation_micro_f1": best_validation["micro_f1"],
    }
    return cpu_state_dict(model), best_epoch, extra


def train_temporal_ssl(
    name: str,
    model: nn.Module,
    graph,
    probe_split: ProbeSplit,
    settings: dict,
    probe_settings: dict,
    seed: int,
    logger,
) -> tuple[dict[str, torch.Tensor], int]:
    """TGN / TGAT / DyGLib backbones: link-prediction pretraining on the
    snapshot edges with probe-based checkpoint selection."""
    model.prepare(graph)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    selection_probe_epochs = int(
        settings.get("selection_probe_epochs", probe_settings.get("selection_probe_epochs", 50))
    )
    probe_hidden_dim = probe_settings.get("probe_hidden_dim")
    best_score, best_epoch, best_state = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        metrics = model.train_epoch(optimizer, grad_clip, seed + epoch)
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            embeddings = _temporal_embeddings(model, seed)
            validation = validation_probe(
                embeddings, graph.labels, probe_split, selection_probe_epochs, seed, probe_hidden_dim
            )
            logger.info(
                json.dumps(
                    {"model": name, "epoch": epoch, "train": metrics, "validation_probe": validation}
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_state = cpu_state_dict(model)
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    return best_state, best_epoch


def train_snapshot_ssl(
    name: str,
    model: SnapshotSSLLinkBaseline,
    graph,
    probe_split: ProbeSplit,
    settings: dict,
    probe_settings: dict,
    seed: int,
    logger,
) -> tuple[dict[str, torch.Tensor], int]:
    """CLDG / MaskDGNN / DVGMAE: snapshot self-supervision with probe-based
    checkpoint selection."""
    optimizer = torch.optim.Adam(
        model.pretrain_parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    selection_probe_epochs = int(
        settings.get("selection_probe_epochs", probe_settings.get("selection_probe_epochs", 50))
    )
    probe_hidden_dim = probe_settings.get("probe_hidden_dim")
    best_score, best_epoch, best_state = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        metrics = model.pretrain_epoch(graph.snapshots, optimizer, grad_clip, seed + epoch)
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode_context(graph.snapshots)
            validation = validation_probe(
                embeddings, graph.labels, probe_split, selection_probe_epochs, seed, probe_hidden_dim
            )
            logger.info(
                json.dumps(
                    {"model": name, "epoch": epoch, "train": metrics, "validation_probe": validation}
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_state = cpu_state_dict(model)
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    return best_state, best_epoch


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_node_model(
    key: str, config: dict, graph: DynamicGraph, seed: int, device: torch.device
) -> nn.Module:
    settings = dict(config.get(key, {}))
    if key in JEPA_MODELS:
        model_cls = DyGWM if key == "dyg_wm" else SGJEPA
        return model_cls(
            feature_dim=graph.feature_dim,
            window_size=int(config["window_size"]),
            **settings,
        ).to(device)
    classes = int(graph.labels.max().item()) + 1
    if key == "evolvegcn_h":
        return EvolveGCNHNodeClassifier(
            feature_dim=graph.feature_dim,
            hidden_dim=int(settings.get("hidden_dim", 128)),
            classes=classes,
            layers=int(settings.get("layers", 2)),
        ).to(device)
    if key == "roland":
        return ROLANDNodeClassifier(
            feature_dim=graph.feature_dim,
            hidden_dim=int(settings.get("hidden_dim", 128)),
            classes=classes,
            layers=int(settings.get("layers", 2)),
            dropout=float(settings.get("dropout", 0.0)),
            bptt_steps=int(settings.get("bptt_steps", 4)),
        ).to(device)
    if key == "tgn":
        return TGNNodeSSL(
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            time_dim=int(settings.get("time_dim", 100)),
            num_layers=int(settings.get("layers", 1)),
            num_heads=int(settings.get("heads", 2)),
            num_neighbors=int(settings.get("num_neighbors", 10)),
            dropout=float(settings.get("dropout", 0.1)),
            batch_size=int(settings.get("batch_size", 100)),
            inference_batch_size=int(settings.get("inference_batch_size", 512)),
            seed=seed,
        ).to(device)
    if key == "tgat":
        return TGATNodeSSL(
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            hidden_dim=int(settings.get("hidden_dim", 100)),
            num_layers=int(settings.get("layers", 2)),
            num_heads=int(settings.get("heads", 2)),
            num_neighbors=int(settings.get("num_neighbors", 20)),
            dropout=float(settings.get("dropout", 0.1)),
            uniform_neighbors=bool(settings.get("uniform_neighbors", False)),
            batch_size=int(settings.get("batch_size", 200)),
            inference_batch_size=int(settings.get("inference_batch_size", 256)),
        ).to(device)
    if key in {"cawn", "tcl", "graphmixer", "dygformer"}:
        return DyGLibStatelessNodeSSL(
            model_name=key,
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            time_dim=int(settings.get("time_dim", 100)),
            num_layers=int(settings.get("layers", 2)),
            num_heads=int(settings.get("heads", 2)),
            num_neighbors=int(settings.get("num_neighbors", 20)),
            dropout=float(settings.get("dropout", 0.1)),
            channel_embedding_dim=int(settings.get("channel_embedding_dim", 50)),
            position_feat_dim=int(settings.get("position_feat_dim", graph.feature_dim)),
            walk_length=int(settings.get("walk_length", 1)),
            num_walk_heads=int(settings.get("num_walk_heads", 8)),
            patch_size=int(settings.get("patch_size", 1)),
            max_input_sequence_length=int(settings.get("max_input_sequence_length", 32)),
            time_gap=int(settings.get("time_gap", 2000)),
            batch_size=int(settings.get("batch_size", 200)),
            inference_batch_size=int(settings.get("inference_batch_size", 256)),
            sample_neighbor_strategy=str(settings.get("sample_neighbor_strategy", "recent")),
            time_scaling_factor=float(settings.get("time_scaling_factor", 0.0)),
            seed=seed,
        ).to(device)
    link_kwargs = dict(
        negative_ratio=1.0,
        max_positive_pairs=None,
        new_edges_only=False,
        undirected=True,
        bipartite_source_count=None,
    )
    if key == "cldg":
        return CLDGLinkBaseline(
            feature_dim=graph.feature_dim,
            hidden_dim=int(settings.get("hidden_dim", 128)),
            embedding_dim=int(settings.get("embedding_dim", 128)),
            num_layers=int(settings.get("layers", 2)),
            dropout=float(settings.get("dropout", 0.0)),
            num_spans=int(settings.get("num_spans", 4)),
            num_views=int(settings.get("num_views", 4)),
            view_strategy=str(settings.get("view_strategy", "sequential")),
            temperature=float(settings.get("temperature", 0.07)),
            contrastive_batch_size=int(settings.get("contrastive_batch_size", 1024)),
            probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
            **link_kwargs,
        ).to(device)
    if key == "maskdgnn":
        return MaskDGNNLinkBaseline(
            feature_dim=graph.feature_dim,
            hidden_dim=int(settings.get("hidden_dim", 64)),
            num_layers=int(settings.get("layers", 2)),
            dropout=float(settings.get("dropout", 0.1)),
            window_size=int(settings.get("window_size", 4)),
            mask_ratio=float(settings.get("mask_ratio", 0.3)),
            dynamics_ratio=float(settings.get("dynamics_ratio", 0.7)),
            dynamics_weight=float(settings.get("dynamics_weight", 1.0)),
            existing_offset=float(settings.get("existing_offset", 2.0)),
            new_node_offset=float(settings.get("new_node_offset", -0.5)),
            pagerank_damping=float(settings.get("pagerank_damping", 0.85)),
            pagerank_steps=int(settings.get("pagerank_steps", 10)),
            pretrain_pair_limit=int(settings.get("pretrain_pair_limit", 4096)),
            probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
            **link_kwargs,
        ).to(device)
    if key == "dvgmae":
        return DVGMAELinkBaseline(
            feature_dim=graph.feature_dim,
            hidden_dim=int(settings.get("hidden_dim", 64)),
            num_layers=int(settings.get("layers", 2)),
            dropout=float(settings.get("dropout", 0.1)),
            window_size=int(settings.get("window_size", 4)),
            mask_ratio=float(settings.get("mask_ratio", 0.3)),
            history_balance=float(settings.get("history_balance", 0.5)),
            kl_weight=float(settings.get("kl_weight", 0.001)),
            feature_weight=float(settings.get("feature_weight", 0.1)),
            pretrain_pair_limit=int(settings.get("pretrain_pair_limit", 4096)),
            probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
            **link_kwargs,
        ).to(device)
    raise ValueError(f"unknown node model: {key}")


def prepare_node_data(config: dict, seed: int, device: torch.device):
    """Load the graph, cut the windows and draw the stratified probe split."""
    graph = load_npz(config["data"]["path"]).to(device)
    if graph.labels is None:
        raise ValueError("node classification requires node labels")
    windows = list(graph.windows(int(config["window_size"])))
    probe_cfg = config.get("probe", {})
    probe_split = stratified_split(
        graph.labels,
        float(probe_cfg.get("train_ratio", 0.6)),
        float(probe_cfg.get("validation_ratio_within_train", 0.1)),
        seed,
    )
    return graph, windows, probe_split


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def aggregate_seed_runs(runs: dict[str, dict]) -> dict:
    first = next(iter(runs.values()))
    aggregate: dict[str, dict] = {}
    for metric in first:
        values = [run[metric] for run in runs.values()]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            array = np.asarray(values, dtype=np.float64)
            aggregate[metric] = {"mean": float(array.mean()), "std": float(array.std(ddof=0))}
    return aggregate


def checkpoint_path(args, seed: int) -> Path:
    return (
        args.save_model_dir
        / args.model_name
        / args.dataset_name
        / f"{args.model_name}_ratio{args.train_ratio:g}_seed{seed}.pkl"
    )


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_name", type=str, default="dblp", choices=NODE_DATASETS)
    parser.add_argument(
        "--model_name", type=str, default="DyG-WM", choices=list(NODE_MODEL_KEYS)
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU index, -1 for CPU, or 'auto'"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/node_classification.yaml")
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="labelled-node training fraction (default: probe.train_ratio, 0.4)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="override the configured seeds"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="override the number of epochs"
    )
    parser.add_argument("--save_model_dir", type=Path, default=Path("saved_models/node"))
    parser.add_argument("--save_result_dir", type=Path, default=Path("saved_results/node"))
    parser.add_argument("--log_dir", type=Path, default=Path("logs/node"))
    return parser.parse_args()


def load_run_config(args) -> tuple[str, dict]:
    key = node_model_key(args.model_name)
    config = load_node_config(args.config, args.dataset_name)
    if args.train_ratio is not None:
        if not 0.0 < args.train_ratio < 1.0:
            raise ValueError("--train_ratio must be in (0, 1)")
        config.setdefault("probe", {})["train_ratio"] = float(args.train_ratio)
    args.train_ratio = float(config["probe"]["train_ratio"])
    if args.num_epochs is not None:
        if key in JEPA_MODELS:
            config["training"]["epochs"] = args.num_epochs
            config["training"]["min_checkpoint_epoch"] = min(
                int(config["training"].get("min_checkpoint_epoch", 1)), args.num_epochs
            )
        else:
            config[key]["epochs"] = args.num_epochs
    return key, config


def main() -> None:
    args = parse_args("Train DyG-WM / baselines for dynamic node classification")
    key, config = load_run_config(args)
    training = dict(config["training"])
    settings = dict(config.get(key, {}))
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [42]))
    device = get_device(args.gpu)
    tag = f"{args.model_name}_ratio{args.train_ratio:g}"

    runs: dict[str, dict] = {}
    for seed in seeds:
        run_name = f"{tag}_seed{seed}"
        logger = create_logger(
            run_name, args.log_dir / args.model_name / args.dataset_name / f"{run_name}.log"
        )
        logger.info(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "model": args.model_name,
                    "seed": seed,
                    "train_ratio": args.train_ratio,
                    "device": device_description(device),
                    "model_config": settings,
                    "training_config": training,
                }
            )
        )
        set_random_seed(seed)
        graph, windows, probe_split = prepare_node_data(config, seed, device)
        torch.manual_seed(seed)
        model = build_node_model(key, config, graph, seed, device)

        best_view = None
        extra: dict = {}
        if key in JEPA_MODELS:
            best_state, best_epoch, best_view = train_jepa(
                args.model_name, model, graph, windows, probe_split, training, seed, logger
            )
        elif key in SUPERVISED_MODELS:
            best_state, best_epoch, extra = train_supervised(
                args.model_name, model, graph, probe_split, settings, logger
            )
        elif key in TEMPORAL_SSL_MODELS:
            best_state, best_epoch = train_temporal_ssl(
                args.model_name, model, graph, probe_split, settings, training, seed, logger
            )
        else:
            best_state, best_epoch = train_snapshot_ssl(
                args.model_name, model, graph, probe_split, settings, training, seed, logger
            )
        model.load_state_dict(best_state)
        checkpoint = checkpoint_path(args, seed)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_view": best_view,
                "extra": extra,
                "dataset_name": args.dataset_name,
                "model_name": args.model_name,
                "train_ratio": args.train_ratio,
                "seed": seed,
            },
            checkpoint,
        )
        logger.info(f"best epoch {best_epoch}; checkpoint saved to {checkpoint}")

        result = final_node_evaluation(
            key, model, graph, probe_split, training, settings, seed, best_view
        )
        result["best_epoch"] = float(best_epoch)
        result.update(extra)
        logger.info(json.dumps({"final": result}))
        runs[str(seed)] = result
        result_file = (
            args.save_result_dir / args.model_name / args.dataset_name / f"{run_name}.json"
        )
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(result, indent=2))
        del model, graph, windows
        release_device_memory(device)

    summary = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "train_ratio": args.train_ratio,
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate_seed_runs(runs),
    }
    summary_file = args.save_result_dir / args.model_name / args.dataset_name / f"{tag}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    macro, micro = summary["aggregate"]["macro_f1"], summary["aggregate"]["micro_f1"]
    print(
        f"{args.model_name} on {args.dataset_name} (train ratio {args.train_ratio:g}) over "
        f"{len(seeds)} seed(s): Macro-F1 {macro['mean']:.4f} +- {macro['std']:.4f}, "
        f"Micro-F1 {micro['mean']:.4f} +- {micro['std']:.4f}"
    )


if __name__ == "__main__":
    main()
