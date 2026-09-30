"""Train DyG-WM or a baseline for dynamic link prediction.

Example:
    python train_link_prediction.py --dataset_name wikipedia --model_name DyG-WM --gpu 0

For every seed the script trains the model with random negatives, selects the
checkpoint on validation AP/AUC, saves it, and reports validation/test
metrics under the requested evaluation negative strategy and setting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.DyGLibAdapter import DyGLibLinkBaseline
from models.DyGWM import DyGWM
from models.EdgeBank import EdgeBankLinkBaseline
from models.JODIE import JODIELinkBaseline
from models.SnapshotSSL import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
    SnapshotSSLLinkBaseline,
)
from models.TGAT import TGATLinkBaseline
from utils.DataLoader import DynamicGraph, load_npz
from utils.inductive_setting import (
    InductiveSetting,
    build_inductive_setting,
    normalize_setting,
)
from utils.link_utils import (
    TemporalWindowSplit,
    negative_edge_table_from_snapshots,
    temporal_window_split,
)
from utils.load_configs import (
    DATASETS,
    MODEL_KEYS,
    SNAPSHOT_SSL_MODELS,
    load_config,
    model_arguments,
    model_key,
    training_arguments,
)
from utils.negative_sampling import NegativeEdgeTable, normalize_negative_strategy
from utils.temporal_utils import unique_snapshots
from utils.utils import (
    cpu_state_dict,
    create_logger,
    device_description,
    get_device,
    release_device_memory,
    set_random_seed,
)

DYGLIB_MODELS = ("jodie", "dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer")
SNAPSHOT_SSL_CLASSES = {
    "cldg": CLDGLinkBaseline,
    "maskdgnn": MaskDGNNLinkBaseline,
    "dvgmae": DVGMAELinkBaseline,
}


# ----------------------------------------------------------------------------
# Data and evaluation protocol
# ----------------------------------------------------------------------------
@dataclass
class LinkData:
    graph: DynamicGraph
    split: TemporalWindowSplit
    link_cfg: dict
    negative_edge_table: NegativeEdgeTable | None
    inductive: InductiveSetting | None

    @property
    def train_snapshots(self):
        return unique_snapshots(self.split.train)


def negative_destination_pool(graph: DynamicGraph) -> torch.Tensor:
    """DyGLib's negative-destination support: every destination of the stream."""
    destinations = [
        snapshot.query_edge_index[1].detach().cpu()
        for snapshot in graph.snapshots
        if snapshot.query_edge_index is not None
        and snapshot.query_edge_index.shape[1] > 0
    ]
    if not destinations:
        raise ValueError("link prediction requires query_edge_index events")
    return torch.unique(torch.cat(destinations), sorted=True)


def build_negative_edge_table(
    graph: DynamicGraph,
    split: TemporalWindowSplit,
    link_cfg: dict,
    config: dict,
    strategy: str,
) -> NegativeEdgeTable | None:
    """DyGLib historical / inductive evaluation negatives.

    As in DyGLib's ``evaluate_link_prediction.py`` the sampler sees the full
    stream, validation uses seed 0 and test seed 2, and batches hold
    ``eval_positive_batch_size`` consecutive positives.  Models are still
    trained and checkpointed with random negatives.
    """
    if strategy == "random":
        return None
    if (
        float(link_cfg.get("negative_ratio", 1.0)) != 1.0
        or link_cfg.get("max_positive_pairs") is not None
        or bool(link_cfg.get("new_edges_only", False))
        or bool(link_cfg.get("undirected", False))
        or link_cfg.get("eval_positive_batch_size") is None
    ):
        raise ValueError(
            f"negative_sample_strategy={strategy} follows DyGLib's protocol and "
            "needs negative_ratio=1.0, max_positive_pairs=null, "
            "new_edges_only=false, undirected=false and eval_positive_batch_size"
        )
    shared_training = dict(config.get("training", {}))
    seeds = {
        "validation": int(shared_training.get("validation_query_seed", 0)),
        "test": int(shared_training.get("test_query_seed", 2)),
    }
    for section_name, section in config.items():
        if not (section_name.endswith("_training") and isinstance(section, dict)):
            continue
        for key, split_name in (
            ("validation_query_seed", "validation"),
            ("test_query_seed", "test"),
        ):
            if key in section and int(section[key]) != seeds[split_name]:
                raise ValueError(
                    f"{section_name}.{key} differs from training.{key}; "
                    f"{strategy} negatives are sampled once and shared by every model"
                )
    return negative_edge_table_from_snapshots(
        graph.snapshots,
        {
            "validation": unique_snapshots(split.validation, targets_only=True),
            "test": unique_snapshots(split.test, targets_only=True),
        },
        seeds,
        strategy=strategy,
        batch_size=int(link_cfg["eval_positive_batch_size"]),
        bipartite_source_count=graph.num_source_nodes,
    )


def prepare_link_data(
    config: dict,
    device: torch.device,
    negative_strategy: str,
    setting: str,
    logger,
) -> LinkData:
    """Load the dataset, split it and build the shared evaluation protocol."""
    graph = load_npz(config["data"]["path"]).to(device)
    split_cfg = config.get("split", {})
    split = temporal_window_split(
        graph.snapshots,
        int(split_cfg["window_size"]),
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("validation_ratio", 0.2)),
    )
    link_cfg = dict(config.get("link", {}))
    inductive: InductiveSetting | None = None
    if setting == "inductive":
        # DyGLib inductive (new-node) setting: 10% of the nodes are removed
        # from training, checkpoints are selected on the transductive
        # validation set, and the reported metrics score only events that
        # touch a node unseen in training.
        if negative_strategy != "random":
            raise ValueError(
                "--setting inductive reports DyGLib's new-node metrics with random "
                "negatives; use --negative_sample_strategy random"
            )
        inductive_cfg = dict(config.get("inductive_setting", {}))
        inductive = build_inductive_setting(
            graph.snapshots,
            split,
            ratio=float(inductive_cfg.get("new_node_ratio", 0.1)),
            seed=int(inductive_cfg.get("new_node_seed", 2020)),
        )
        logger.info(json.dumps({"inductive_setting": inductive.summary}))
        split = TemporalWindowSplit(
            train=inductive.train_windows,
            validation=split.validation,
            test=split.test,
        )
    link_cfg["negative_destination_candidates"] = negative_destination_pool(graph)
    configured = link_cfg.get("bipartite_source_count")
    if configured is not None and (
        graph.num_source_nodes is None or int(configured) != graph.num_source_nodes
    ):
        raise ValueError("configured bipartite split disagrees with the dataset")
    link_cfg["bipartite_source_count"] = graph.num_source_nodes
    negative_edge_table = build_negative_edge_table(
        graph, split, link_cfg, config, negative_strategy
    )
    if negative_edge_table is not None:
        logger.info(
            json.dumps(
                {
                    "negative_strategy": negative_strategy,
                    "negative_edges": negative_edge_table.summary,
                }
            )
        )
    return LinkData(graph, split, link_cfg, negative_edge_table, inductive)


def _target_event_count(windows: list[list]) -> int | None:
    targets = unique_snapshots(windows, targets_only=True)
    if any(snapshot.query_edge_index is None for snapshot in targets):
        return None
    return sum(int(snapshot.query_edge_index.shape[1]) for snapshot in targets)


def assert_full_event_coverage(
    model_name: str, result: dict, split: TemporalWindowSplit, link_cfg: dict
) -> None:
    """Fail if a model silently dropped evaluation events (1:1 protocol)."""
    if (
        float(link_cfg.get("negative_ratio", 1.0)) != 1.0
        or link_cfg.get("max_positive_pairs") is not None
        or bool(link_cfg.get("new_edges_only", False))
    ):
        return
    for split_name, windows in (("validation", split.validation), ("test", split.test)):
        event_count = _target_event_count(windows)
        if event_count is None or split_name not in result:
            continue
        expected = float(2 * event_count)
        actual = float(result[split_name]["examples"])
        if actual != expected:
            raise RuntimeError(
                f"{model_name} evaluated {actual:g} {split_name} examples; the "
                f"1:1 protocol requires {expected:g} from all target events"
            )


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_model(key: str, config: dict, data: LinkData, device: torch.device) -> nn.Module:
    """Construct model ``key`` and attach its data-dependent state."""
    graph, link_cfg, inductive = data.graph, data.link_cfg, data.inductive
    arguments = model_arguments(config, key)
    runtime = {
        "negative_destination_candidates": link_cfg["negative_destination_candidates"],
        "bipartite_source_count": link_cfg["bipartite_source_count"],
    }
    if key == "dyg_wm":
        model = DyGWM(
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            **{**arguments, **runtime},
        ).to(device)
        # Inductive setting: the causal event history is built from the
        # reduced training snapshots followed by the full validation/test
        # ones, so no held-out training event reaches a training query.
        model.prepare_causal_history(
            graph.snapshots if inductive is None else inductive.history_snapshots
        )
        return model
    if key == "tgat":
        model = TGATLinkBaseline(
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            **{**arguments, **runtime},
        ).to(device)
        model.prepare_streams(graph.snapshots, data.train_snapshots)
        return model
    if key == "jodie_bipartite":
        if graph.num_source_nodes is None:
            raise ValueError("JODIE-Bipartite requires a bipartite user-item graph")
        model = JODIELinkBaseline(
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            **{**arguments, **runtime},
        ).to(device)
        # The original implementation standardises event gaps and chooses the
        # t-batch span from the complete stream (a preprocessing detail, not a
        # learned use of validation/test labels).
        model.fit_stream_statistics(
            graph.snapshots, JODIELinkBaseline._unique_snapshots(data.split.train)
        )
        return model
    if key in SNAPSHOT_SSL_CLASSES:
        return SNAPSHOT_SSL_CLASSES[key](
            feature_dim=graph.feature_dim, **{**arguments, **runtime}
        ).to(device)
    if key == "edgebank":
        model = EdgeBankLinkBaseline(
            num_nodes=graph.num_nodes, **{**arguments, **runtime}
        ).to(device)
        model.eval()
        return model
    if key in DYGLIB_MODELS:
        model = DyGLibLinkBaseline(
            model_name=key,
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            **{**arguments, **runtime},
        ).to(device)
        model.prepare_streams(
            graph.snapshots,
            data.train_snapshots,
            exclude_nodes=None if inductive is None else inductive.sampled_nodes.numpy(),
        )
        return model
    raise ValueError(f"unknown model {key}")


def is_event_model(model: nn.Module) -> bool:
    """JODIE, TGAT, EdgeBank and the DyGLib backbones use their native
    event-stream evaluation."""
    return isinstance(
        model,
        (JODIELinkBaseline, TGATLinkBaseline, DyGLibLinkBaseline, EdgeBankLinkBaseline),
    )


def evaluate_windows(
    model: nn.Module,
    windows: list[list],
    history_windows: list[list],
    query_seed: int,
    pair_batch_size: int | None,
) -> dict[str, float]:
    if is_event_model(model):
        return model.evaluate_protocol(  # type: ignore[attr-defined]
            windows, history_windows, query_seed=query_seed
        )
    return model.evaluate_windows(  # type: ignore[attr-defined]
        windows, pair_batch_size=pair_batch_size, query_seed=query_seed
    )


def _attach_inductive_evaluation(
    model: nn.Module, inductive: InductiveSetting, split_name: str
) -> None:
    """Point the evaluator at the new-node destination pool of ``split_name``."""
    pool = inductive.validation_pool if split_name == "validation" else inductive.test_pool
    model.negative_destination_candidates = pool  # type: ignore[attr-defined]
    if isinstance(model, DyGLibLinkBaseline):
        model.inductive_evaluation = (
            inductive.new_node_ids,
            pool.detach().cpu().numpy().astype(np.int64),
        )


def final_evaluation(
    model: nn.Module, data: LinkData, training: dict
) -> tuple[dict[str, float], dict[str, float]]:
    """Validation/test metrics of the selected checkpoint under the requested
    negative strategy (and new-node setting)."""
    split, inductive = data.split, data.inductive
    pair_batch_size = training.get("pair_batch_size")
    if isinstance(model, SnapshotSSLLinkBaseline):
        pair_batch_size = int(training.get("pair_batch_size", 512))
    validation_query_seed = int(training.get("validation_query_seed", 0))
    test_query_seed = int(training.get("test_query_seed", 2))
    model.eval()
    if data.negative_edge_table is not None:
        model.negative_edge_table = data.negative_edge_table  # type: ignore[attr-defined]
    if inductive is None:
        validation = evaluate_windows(
            model, split.validation, split.train, validation_query_seed, pair_batch_size
        )
        test = evaluate_windows(
            model,
            split.test,
            [*split.train, *split.validation],
            test_query_seed,
            pair_batch_size,
        )
    else:
        _attach_inductive_evaluation(model, inductive, "validation")
        validation = evaluate_windows(
            model,
            inductive.validation_windows,
            inductive.full_train_windows,
            inductive.validation_query_seed,
            pair_batch_size,
        )
        _attach_inductive_evaluation(model, inductive, "test")
        test = evaluate_windows(
            model,
            inductive.test_windows,
            [*inductive.full_train_windows, *split.validation],
            inductive.test_query_seed,
            pair_batch_size,
        )
    return validation, test


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train_model(
    name: str,
    model: nn.Module,
    data: LinkData,
    training: dict,
    seed: int,
    logger,
) -> tuple[dict[str, torch.Tensor], int]:
    """Train with random negatives and return the best checkpoint and its epoch."""
    split = data.split
    event_model = is_event_model(model)
    betas = tuple(float(value) for value in training.get("betas", (0.9, 0.999)))
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if event_model:
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=betas,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=(0.9, 0.95),
            eps=1e-10,
        )
    scheduler = None
    if "lr_reduce_factor" in training:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training["lr_reduce_factor"]),
            patience=int(training.get("lr_patience", 5)),
            min_lr=float(training.get("min_learning_rate", 0.0)),
        )
    epochs = int(training["epochs"])
    pair_batch_size = training.get("pair_batch_size")
    eval_every = int(training.get("eval_every", 1))
    checkpoint_metrics = tuple(training.get("checkpoint_metrics", ("ap",)))
    if not checkpoint_metrics:
        raise ValueError("checkpoint_metrics must not be empty")
    best_metrics = {metric: float("-inf") for metric in checkpoint_metrics}
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = int(training.get("patience", 0))
    evaluations_without_improvement = 0
    validation_query_seed = int(training.get("validation_query_seed", 0))

    def validate() -> dict[str, float]:
        return evaluate_windows(
            model, split.validation, split.train, validation_query_seed, pair_batch_size
        )

    if bool(training.get("evaluate_before_training", False)):
        model.eval()
        validation = validate()
        logger.info(json.dumps({"model": name, "epoch": 0, "validation": validation}))
        best_metrics = {metric: validation[metric] for metric in checkpoint_metrics}
        best_epoch = 0
        best_state = cpu_state_dict(model)

    def train_step(epoch: int) -> tuple[dict, float]:
        model.train()
        if isinstance(model, JODIELinkBaseline):
            metrics = model.train_epoch(
                split.train, optimizer, float(training["grad_clip"])
            )
        elif isinstance(model, DyGLibLinkBaseline):
            metrics = model.train_epoch(
                split.train, optimizer, float(training["grad_clip"]), seed=seed
            )
        elif isinstance(model, TGATLinkBaseline):
            metrics = model.train_epoch(
                split.train,
                optimizer,
                float(training["grad_clip"]),
                seed=seed + epoch * 10_000,
            )
        else:
            metrics = model.train_epoch(  # type: ignore[attr-defined]
                split.train,
                optimizer,
                float(training["grad_clip"]),
                seed=seed + epoch * 10_000,
                pair_batch_size=pair_batch_size,
            )
        # DyG-WM updates its EMA targets inside train_epoch when
        # ema_update_per_step / ema_steps_per_epoch is set; otherwise once
        # per epoch here.
        if hasattr(model, "update_target_encoder") and not (
            isinstance(model, DyGWM)
            and (model.ema_update_per_step or model.ema_steps_per_epoch is not None)
        ):
            model.update_target_encoder()
        return metrics, float(metrics["loss"])

    for epoch in range(1, epochs + 1):
        metrics, loss_value = train_step(epoch)
        if not np.isfinite(loss_value):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            validation = validate()
            if scheduler is not None:
                scheduler.step(validation["ap"])
            logger.info(
                json.dumps(
                    {"model": name, "epoch": epoch, "train": metrics, "validation": validation}
                )
            )
            if all(
                validation[metric] >= best_metrics[metric] for metric in checkpoint_metrics
            ):
                best_metrics = {metric: validation[metric] for metric in checkpoint_metrics}
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if patience > 0 and evaluations_without_improvement >= patience:
                    logger.info(f"early stopping at epoch {epoch}")
                    break

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    return best_state, best_epoch


def train_snapshot_ssl(
    name: str,
    model: SnapshotSSLLinkBaseline,
    data: LinkData,
    training: dict,
    seed: int,
    logger,
) -> tuple[dict[str, torch.Tensor], int]:
    """Snapshot SSL baselines: native self-supervised pretraining on the
    training snapshots, then a frozen-encoder pair-MLP link probe selected on
    validation.  Returns the full model state with the selected probe."""
    split = data.split
    pretrain_optimizer = torch.optim.Adam(
        model.pretrain_parameters(),
        lr=float(training["pretrain_learning_rate"]),
        weight_decay=float(training.get("pretrain_weight_decay", 0.0)),
    )
    train_snapshots = unique_snapshots(split.train)
    pretrain_epochs = int(training["pretrain_epochs"])
    grad_clip = float(training.get("grad_clip", 1.0))
    for epoch in range(1, pretrain_epochs + 1):
        metrics = model.pretrain_epoch(
            train_snapshots, pretrain_optimizer, grad_clip, seed + epoch * 10_000
        )
        if not np.isfinite(float(metrics["loss"])):
            raise RuntimeError(f"{name} produced a non-finite SSL loss at epoch {epoch}")
        if (
            epoch == 1
            or epoch % int(training.get("pretrain_log_every", 10)) == 0
            or epoch == pretrain_epochs
        ):
            logger.info(
                json.dumps({"model": name, "stage": "ssl_pretrain", "epoch": epoch, "train": metrics})
            )

    model.freeze_encoder()
    model.eval()
    model.probe.train()
    probe_optimizer = torch.optim.Adam(
        model.probe.parameters(),
        lr=float(training["probe_learning_rate"]),
        weight_decay=float(training.get("probe_weight_decay", 0.0)),
    )
    probe_epochs = int(training["probe_epochs"])
    eval_every = int(training.get("eval_every", 1))
    pair_batch_size = int(training.get("pair_batch_size", 512))
    patience = int(training.get("patience", 0))
    validation_query_seed = int(training.get("validation_query_seed", 0))
    stale_evaluations = 0
    checkpoint_metrics = tuple(training.get("checkpoint_metrics", ("ap",)))
    if not checkpoint_metrics:
        raise ValueError("checkpoint_metrics must not be empty")
    best_metrics = {metric: float("-inf") for metric in checkpoint_metrics}
    best_epoch = 0
    best_probe_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, probe_epochs + 1):
        metrics = model.train_probe_epoch(
            split.train, probe_optimizer, grad_clip, pair_batch_size, seed + epoch * 10_000
        )
        if epoch == 1 or epoch % eval_every == 0 or epoch == probe_epochs:
            validation = model.evaluate_windows(
                split.validation,
                pair_batch_size=pair_batch_size,
                query_seed=validation_query_seed,
            )
            logger.info(
                json.dumps(
                    {
                        "model": name,
                        "stage": "frozen_link_probe",
                        "epoch": epoch,
                        "train": metrics,
                        "validation": validation,
                    }
                )
            )
            if all(
                validation[metric] >= best_metrics[metric] for metric in checkpoint_metrics
            ):
                best_metrics = {metric: validation[metric] for metric in checkpoint_metrics}
                best_epoch = epoch
                best_probe_state = cpu_state_dict(model.probe)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if patience > 0 and stale_evaluations >= patience:
                break
    if best_probe_state is None:
        raise RuntimeError(f"{name} did not produce a frozen-probe checkpoint")
    model.probe.load_state_dict(best_probe_state)
    return cpu_state_dict(model), best_epoch


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def result_suffix(negative_strategy: str, setting: str) -> str:
    suffix = "" if negative_strategy == "random" else f"_{negative_strategy}"
    return suffix + ("_inductive_setting" if setting == "inductive" else "")


def aggregate_seed_runs(runs: dict[str, dict]) -> dict:
    """Mean / population std of every numeric metric over seeds."""
    first = next(iter(runs.values()))
    aggregate: dict[str, dict] = {}
    for split_name in ("validation", "test"):
        aggregate[split_name] = {}
        for metric in first[split_name]:
            values = [run[split_name][metric] for run in runs.values()]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                array = np.asarray(values, dtype=np.float64)
                aggregate[split_name][metric] = {
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=0)),
                }
    return aggregate


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_name", type=str, default="wikipedia", choices=DATASETS)
    parser.add_argument("--model_name", type=str, default="DyG-WM", choices=list(MODEL_KEYS))
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU index, -1 for CPU, or 'auto'"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/link_prediction.yaml"))
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive"],
        help="evaluation negatives (training always uses random negatives)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="transductive",
        choices=["transductive", "inductive"],
        help="'inductive' holds out 10%% of the nodes from training (DyGLib new-node setting)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="override the configured seeds"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="override the number of epochs"
    )
    parser.add_argument("--save_model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--save_result_dir", type=Path, default=Path("saved_results"))
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args("Train DyG-WM / baselines for dynamic link prediction")
    key = model_key(args.model_name)
    negative_strategy = normalize_negative_strategy(args.negative_sample_strategy)
    setting = normalize_setting(args.setting)
    config = load_config(args.config, args.dataset_name)
    if args.num_epochs is not None:
        section = config.setdefault(f"{key}_training", {})
        if key in SNAPSHOT_SSL_MODELS:
            section["pretrain_epochs"] = args.num_epochs
            section["probe_epochs"] = args.num_epochs
        else:
            section["epochs"] = args.num_epochs
    training = training_arguments(config, key)
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [0]))
    device = get_device(args.gpu)
    suffix = result_suffix(negative_strategy, setting)
    model_dir = args.model_name

    runs: dict[str, dict] = {}
    for seed in seeds:
        run_name = f"{model_dir}_seed{seed}{suffix}"
        logger = create_logger(
            run_name, args.log_dir / model_dir / args.dataset_name / f"{run_name}.log"
        )
        logger.info(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "model": args.model_name,
                    "seed": seed,
                    "negative_sample_strategy": negative_strategy,
                    "setting": setting,
                    "device": device_description(device),
                    "model_config": model_arguments(config, key),
                    "training_config": training,
                }
            )
        )
        set_random_seed(seed)
        data = prepare_link_data(config, device, negative_strategy, setting, logger)

        if key != "edgebank":
            torch.manual_seed(seed)
        model = build_model(key, config, data, device)
        logger.info(
            f"trainable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        best_epoch = 0
        if key != "edgebank":
            if key in SNAPSHOT_SSL_MODELS:
                best_state, best_epoch = train_snapshot_ssl(
                    args.model_name, model, data, training, seed, logger
                )
            else:
                best_state, best_epoch = train_model(
                    args.model_name, model, data, training, seed, logger
                )
            model.load_state_dict(best_state)
            checkpoint = (
                args.save_model_dir / model_dir / args.dataset_name / f"{model_dir}_seed{seed}"
                f"{'_inductive_setting' if setting == 'inductive' else ''}.pkl"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "dataset_name": args.dataset_name,
                    "model_name": args.model_name,
                    "seed": seed,
                    "setting": setting,
                },
                checkpoint,
            )
            logger.info(f"best epoch {best_epoch}; checkpoint saved to {checkpoint}")

        validation, test = final_evaluation(model, data, training)
        test["best_epoch"] = float(best_epoch)
        if key in SNAPSHOT_SSL_MODELS:
            test.update(
                {
                    "pretrain_epochs": float(training["pretrain_epochs"]),
                    "protocol": "ssl_pretrain_then_frozen_link_probe",
                    "implementation": model.implementation,
                }
            )
        result = {"validation": validation, "test": test}
        assert_full_event_coverage(
            args.model_name,
            result,
            data.split if data.inductive is None else data.inductive.final_split(data.split),
            data.link_cfg,
        )
        logger.info(json.dumps({"final": result}))
        runs[str(seed)] = result

        result_file = (
            args.save_result_dir / model_dir / args.dataset_name / f"{run_name}.json"
        )
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(result, indent=2))
        del model, data
        release_device_memory(device)

    summary = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "negative_sample_strategy": negative_strategy,
        "setting": setting,
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate_seed_runs(runs),
    }
    summary_file = (
        args.save_result_dir / model_dir / args.dataset_name / f"{model_dir}{suffix}.json"
    )
    summary_file.write_text(json.dumps(summary, indent=2))
    test_summary = summary["aggregate"]["test"]
    print(
        f"{args.model_name} on {args.dataset_name} ({negative_strategy}, {setting}) "
        f"over {len(seeds)} seed(s): test AP {test_summary['ap']['mean']:.4f} "
        f"+- {test_summary['ap']['std']:.4f}, AUC {test_summary['auc']['mean']:.4f} "
        f"+- {test_summary['auc']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
