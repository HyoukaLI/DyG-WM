"""EdgeBank (Poursafaei et al., NeurIPS 2022) under the shared event protocol."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from utils.DataLoader import Snapshot
from utils.temporal_utils import SharedLinkProtocol, unique_snapshots


class EdgeBankLinkBaseline(nn.Module, SharedLinkProtocol):
    """EdgeBank baseline under the shared event protocol."""

    def __init__(
        self,
        num_nodes: int,
        bipartite_source_count: int | None = None,
        negative_ratio: float = 20.0,
        max_positive_pairs: int | None = 1024,
        new_edges_only: bool = False,
        undirected: bool = False,
        negative_destination_candidates: Tensor | None = None,
        allow_negative_collisions: bool = False,
        eval_positive_batch_size: int | None = None,
        memory_mode: str = "time_window_memory",
        time_window_proportion: float = 0.15,
    ) -> None:
        super().__init__()
        if undirected:
            raise ValueError("event-stream comparison requires directed links")
        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only
        self.negative_destination_candidates = negative_destination_candidates
        self.allow_negative_collisions = allow_negative_collisions
        self.eval_positive_batch_size = eval_positive_batch_size
        if memory_mode not in {"unlimited_memory", "time_window_memory"}:
            raise ValueError("unsupported EdgeBank memory mode")
        if not 0.0 < time_window_proportion <= 1.0:
            raise ValueError("time_window_proportion must be in (0, 1]")
        self.memory_mode = memory_mode
        self.time_window_proportion = time_window_proportion

    @staticmethod
    def _pairs(snapshot: Snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if snapshot.query_edge_index is None:
            raise ValueError("EdgeBank requires event query edges")
        sources, destinations = snapshot.query_edge_index.detach().cpu().numpy()
        if snapshot.query_timestamps is None:
            times = np.full(len(sources), float(snapshot.time), dtype=np.float64)
        else:
            times = snapshot.query_timestamps.detach().cpu().numpy().astype(np.float64)
        order = np.argsort(times, kind="stable")
        return sources[order], destinations[order], times[order]

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        if (
            self.negative_ratio == 1.0
            and self.max_positive_pairs is None
            and self.eval_positive_batch_size is not None
        ):
            history_snapshots = unique_snapshots(history_windows)
            target_snapshots = unique_snapshots(windows, targets_only=True)
            history_entries = [self._pairs(snapshot) for snapshot in history_snapshots]
            target_entries = [self._pairs(snapshot) for snapshot in target_snapshots]
            history_sources = np.concatenate([entry[0] for entry in history_entries])
            history_destinations = np.concatenate([entry[1] for entry in history_entries])
            history_times = np.concatenate([entry[2] for entry in history_entries])
            target_sources = np.concatenate([entry[0] for entry in target_entries])
            target_destinations = np.concatenate([entry[1] for entry in target_entries])
            target_times = np.concatenate([entry[2] for entry in target_entries])
            pool = self.negative_destination_candidates
            if pool is None:
                pool = torch.unique(
                    torch.as_tensor(
                        np.concatenate([history_destinations, target_destinations])
                    ),
                    sorted=True,
                )
            pool_array = pool.detach().cpu().numpy().astype(np.int64)
            table_negatives = None
            if self.negative_edge_table is not None:
                table_negatives = self.negative_edge_table.for_snapshots(
                    [snapshot.time for snapshot in target_snapshots]
                )
                table_negatives.check_alignment(
                    target_sources, target_destinations, context="EdgeBank evaluation"
                )
            rng = np.random.RandomState(query_seed)
            scores: list[Tensor] = []
            labels: list[Tensor] = []
            groups: list[Tensor] = []
            group_offset = 0
            batch_size = self.eval_positive_batch_size
            for start in range(0, len(target_sources), batch_size):
                stop = min(start + batch_size, len(target_sources))
                if self.memory_mode == "time_window_memory":
                    threshold = np.quantile(
                        history_times, 1.0 - self.time_window_proportion
                    )
                    keep = history_times >= threshold
                else:
                    keep = np.ones(len(history_times), dtype=bool)
                seen = set(
                    zip(
                        history_sources[keep].tolist(),
                        history_destinations[keep].tolist(),
                    )
                )
                source = target_sources[start:stop]
                positive = target_destinations[start:stop]
                if table_negatives is None:
                    negative_source = source
                    negative = rng.choice(pool_array, size=len(source), replace=True)
                else:
                    negative_source = table_negatives.sources[start:stop]
                    negative = table_negatives.destinations[start:stop]
                positive_scores = torch.tensor(
                    [float((int(u), int(v)) in seen) for u, v in zip(source, positive)]
                )
                negative_scores = torch.tensor(
                    [
                        float((int(u), int(v)) in seen)
                        for u, v in zip(negative_source, negative)
                    ]
                )
                scores.extend([positive_scores, negative_scores])
                labels.extend(
                    [torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]
                )
                group = torch.arange(group_offset, group_offset + len(source))
                groups.extend([group, group])
                group_offset += len(source)
                history_sources = np.concatenate([history_sources, source])
                history_destinations = np.concatenate(
                    [history_destinations, positive]
                )
                history_times = np.concatenate(
                    [history_times, target_times[start:stop]]
                )
            return self.metrics(
                torch.cat(labels), torch.cat(scores), torch.cat(groups)
            )

        seen: set[tuple[int, int]] = set()
        for snapshot in unique_snapshots(history_windows):
            sources, destinations, _ = self._pairs(snapshot)
            seen.update(zip(sources.tolist(), destinations.tolist()))
        score_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            sources, destinations, event_times = self._pairs(window[-1])
            event_cursor = 0
            query_times = (
                queries.timestamps
                if queries.timestamps is not None
                else queries.labels.new_full(queries.labels.shape, float(window[-1].time))
            )
            scores = torch.empty_like(queries.labels)
            for timestamp in sorted(set(query_times.detach().cpu().tolist())):
                before = int(np.searchsorted(event_times, timestamp, side="left"))
                seen.update(
                    zip(
                        sources[event_cursor:before].tolist(),
                        destinations[event_cursor:before].tolist(),
                    )
                )
                event_cursor = before
                rows = torch.nonzero(query_times == timestamp, as_tuple=False).flatten()
                for row in rows.detach().cpu().tolist():
                    pair = tuple(queries.pairs[row].detach().cpu().tolist())
                    scores[row] = float(pair in seen)
                through = int(np.searchsorted(event_times, timestamp, side="right"))
                seen.update(
                    zip(
                        sources[event_cursor:through].tolist(),
                        destinations[event_cursor:through].tolist(),
                    )
                )
                event_cursor = through
            seen.update(zip(sources[event_cursor:].tolist(), destinations[event_cursor:].tolist()))
            score_parts.append(scores)
            label_parts.append(queries.labels)
            group_parts.append(queries.group_ids + group_offset)
            group_offset += int(queries.group_ids.max().item()) + 1
        return self.metrics(
            torch.cat(label_parts), torch.cat(score_parts), torch.cat(group_parts)
        )
