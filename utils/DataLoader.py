from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


@dataclass
class Snapshot:
    """A discrete-time graph snapshot with a shared global node id space."""

    x: Tensor
    edge_index: Tensor
    active: Tensor
    time: int
    # Optional directed event edges used as link-prediction queries.  Unlike
    # edge_index, duplicates are meaningful and are intentionally preserved.
    query_edge_index: Tensor | None = None
    # Optional per-event information aligned column-wise with query_edge_index.
    # Event-stream baselines (TGAT, DyGLib models) and DyG-WM's causal history
    # read the exact timestamps and event features.
    query_timestamps: Tensor | None = None
    query_features: Tensor | None = None
    # Optional per-event state-change label from the raw data (unused here).
    query_labels: Tensor | None = None

    def validate(self, num_nodes: int) -> None:
        if self.x.ndim != 2 or self.x.shape[0] != num_nodes:
            raise ValueError("x must have shape [num_nodes, feature_dim]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.query_edge_index is not None and (
            self.query_edge_index.ndim != 2 or self.query_edge_index.shape[0] != 2
        ):
            raise ValueError("query_edge_index must have shape [2, num_events]")
        num_events = 0 if self.query_edge_index is None else self.query_edge_index.shape[1]
        if self.query_timestamps is not None and self.query_timestamps.shape != (num_events,):
            raise ValueError("query_timestamps must align with query_edge_index")
        if self.query_features is not None and (
            self.query_features.ndim != 2 or self.query_features.shape[0] != num_events
        ):
            raise ValueError("query_features must have shape [num_events, feature_dim]")
        if self.query_labels is not None and self.query_labels.shape != (num_events,):
            raise ValueError("query_labels must align with query_edge_index")
        if self.active.shape != (num_nodes,) or self.active.dtype != torch.bool:
            raise ValueError("active must be a boolean [num_nodes] mask")
        if self.edge_index.numel() and (
            self.edge_index.min() < 0 or self.edge_index.max() >= num_nodes
        ):
            raise ValueError("edge_index contains an invalid node id")
        if self.query_edge_index is not None and self.query_edge_index.numel() and (
            self.query_edge_index.min() < 0
            or self.query_edge_index.max() >= num_nodes
        ):
            raise ValueError("query_edge_index contains an invalid node id")


@dataclass
class DynamicGraph:
    snapshots: list[Snapshot]
    labels: Tensor | None = None
    # For bipartite graphs, sources occupy [0, num_source_nodes) and
    # destinations occupy [num_source_nodes, num_nodes).
    num_source_nodes: int | None = None

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("at least one snapshot is required")
        n = self.snapshots[0].x.shape[0]
        d = self.snapshots[0].x.shape[1]
        for snapshot in self.snapshots:
            snapshot.validate(n)
            if snapshot.x.shape[1] != d:
                raise ValueError("all snapshots must share feature_dim")
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError("labels must have shape [num_nodes]")
        if self.num_source_nodes is not None and not 0 < self.num_source_nodes < n:
            raise ValueError("num_source_nodes must split the global node id space")

    @property
    def num_nodes(self) -> int:
        return self.snapshots[0].x.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.snapshots[0].x.shape[1]

    def to(self, device: torch.device | str) -> "DynamicGraph":
        return DynamicGraph(
            [
                Snapshot(
                    s.x.to(device),
                    s.edge_index.to(device),
                    s.active.to(device),
                    s.time,
                    None
                    if s.query_edge_index is None
                    else s.query_edge_index.to(device),
                    None
                    if s.query_timestamps is None
                    else s.query_timestamps.to(device),
                    None
                    if s.query_features is None
                    else s.query_features.to(device),
                    None if s.query_labels is None else s.query_labels.to(device),
                )
                for s in self.snapshots
            ],
            None if self.labels is None else self.labels.to(device),
            self.num_source_nodes,
        )

    def windows(self, window_size: int) -> Iterable[list[Snapshot]]:
        """Non-overlapping windows of ``window_size`` snapshots (node
        classification); an incomplete tail is dropped."""
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        usable = len(self.snapshots) // window_size * window_size
        for start in range(0, usable, window_size):
            yield self.snapshots[start : start + window_size]


def load_npz(path: str | Path) -> DynamicGraph:
    """Load a processed dynamic graph archive (``processed_data/<name>.npz``).

    Expected keys: ``features`` [T,N,F] (per-snapshot node features),
    ``edges_<t>`` [2,E_t] (structural edges of snapshot t), optional
    ``queries_<t>`` [2,Q_t] with aligned ``query_timestamps_<t>``,
    ``query_features_<t>`` and ``query_labels_<t>`` (the raw events of
    snapshot t, duplicates kept), optional ``active`` [T,N], ``labels`` [N]
    and ``num_source_nodes`` (bipartite graphs). Object arrays are accepted
    for variable-size edge lists, hence allow_pickle is required.
    """
    raw = np.load(Path(path), allow_pickle=True)
    features = raw["features"]
    if "feature_source" in raw and str(raw["feature_source"]) == "structural-fallback":
        message = (
            f"{path} carries the 4-D structural fallback instead of the SpikeNet "
            "DeepWalk features; node-classification results will NOT match the "
            "SpikeNet / SG-JEPA protocol (every model degenerates to the majority "
            "class). Rebuild the archive with the <dataset>.npy features (see "
            "DG_data/DATASETS_README.md). Set DYGWM_ALLOW_STRUCTURAL_FALLBACK=1 to "
            "run the fallback protocol on purpose."
        )
        if os.environ.get("DYGWM_ALLOW_STRUCTURAL_FALLBACK", "0") != "1":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if features.ndim == 3 and any(not np.any(features[t]) for t in range(features.shape[0])):
        raise RuntimeError(
            f"{path}: at least one snapshot has all-zero node features; the archive "
            "is incomplete; rebuild it with the scripts in preprocess_data/."
        )
    active = raw["active"] if "active" in raw else np.ones(features.shape[:2], bool)
    snapshots = []
    for t in range(features.shape[0]):
        edge = np.asarray(raw[f"edges_{t}"], dtype=np.int64)
        snapshots.append(
            Snapshot(
                x=torch.as_tensor(features[t], dtype=torch.float32),
                edge_index=torch.as_tensor(edge, dtype=torch.long),
                active=torch.as_tensor(active[t], dtype=torch.bool),
                time=t,
                query_edge_index=(
                    torch.as_tensor(
                        np.asarray(raw[f"queries_{t}"], dtype=np.int64),
                        dtype=torch.long,
                    )
                    if f"queries_{t}" in raw
                    else None
                ),
                query_timestamps=(
                    torch.as_tensor(
                        np.asarray(raw[f"query_timestamps_{t}"], dtype=np.float64),
                        dtype=torch.float32,
                    )
                    if f"query_timestamps_{t}" in raw
                    else None
                ),
                query_features=(
                    torch.as_tensor(
                        np.asarray(raw[f"query_features_{t}"], dtype=np.float32),
                        dtype=torch.float32,
                    )
                    if f"query_features_{t}" in raw
                    else None
                ),
                query_labels=(
                    torch.as_tensor(
                        np.asarray(raw[f"query_labels_{t}"], dtype=np.int64),
                        dtype=torch.long,
                    )
                    if f"query_labels_{t}" in raw
                    else None
                ),
            )
        )
    labels = torch.as_tensor(raw["labels"], dtype=torch.long) if "labels" in raw else None
    num_source_nodes = (
        int(np.asarray(raw["num_source_nodes"]).item())
        if "num_source_nodes" in raw
        else None
    )
    return DynamicGraph(snapshots, labels, num_source_nodes)

