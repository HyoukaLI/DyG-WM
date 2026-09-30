"""Building blocks of DyG-WM: snapshot encoder, positional/time encodings and
the truncated path signature used by the pair-path and node-path branches."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Mean-aggregation GraphSAGE layer on a sparse edge list."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        if edge_index.numel() == 0:
            mean = torch.zeros_like(x)
        else:
            src, dst = edge_index
            mean = torch.zeros_like(x)
            mean.index_add_(0, dst, x[src])
            degree = torch.bincount(dst, minlength=n).to(x.dtype).clamp_min_(1.0)
            mean = mean / degree.unsqueeze(-1)
        return self.linear(torch.cat([x, mean], dim=-1))


class GraphSAGE(nn.Module):
    """Stack of GraphSAGE layers with ReLU between layers (snapshot encoder)."""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GraphSAGE requires at least one layer")
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(GraphSAGELayer(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i + 1 < len(self.layers):
                x = F.relu(x)
        return x


def sinusoidal_time_encoding(time: int, dim: int, device: torch.device) -> Tensor:
    """Deterministic sinusoidal encoding of one snapshot index / horizon."""
    if dim <= 0:
        return torch.empty(0, device=device)
    positions = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    rates = torch.exp(-math.log(10_000.0) * positions / max(dim, 1))
    out = torch.zeros(dim, device=device)
    out[0::2] = torch.sin(float(time) * rates)
    if dim > 1:
        out[1::2] = torch.cos(float(time) * rates[: out[1::2].numel()])
    return out


def random_walk_positional_encoding(
    edge_index: Tensor, num_nodes: int, dim: int, walks: int = 16, seed: int = 42
) -> Tensor:
    """Monte-Carlo random-walk positional encoding without an N x N matrix.

    Each feature estimates ``diag(P^k)`` from ``walks`` paths per node, using a
    private seeded generator; isolated nodes receive zero encodings.
    """
    device = edge_index.device
    if dim <= 0:
        return torch.empty(num_nodes, 0, device=device)
    if edge_index.numel() == 0 or walks <= 0:
        return torch.zeros(num_nodes, dim, device=device)
    src, dst = edge_index
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    degree = torch.bincount(src, minlength=num_nodes)
    offsets = torch.zeros(num_nodes, dtype=torch.long, device=device)
    offsets[1:] = degree.cumsum(0)[:-1]
    start = torch.arange(num_nodes, device=device).repeat_interleave(walks)
    current = start.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    features = []
    for _ in range(dim):
        deg = degree[current]
        movable = deg > 0
        choice = (
            torch.rand(current.numel(), device=device, generator=generator)
            * deg.clamp_min(1)
        ).long()
        next_node = current.clone()
        next_node[movable] = dst[offsets[current[movable]] + choice[movable]]
        current = next_node
        returned = (current == start).view(num_nodes, walks).float().mean(dim=1)
        returned[degree == 0] = 0
        features.append(returned)
    return torch.stack(features, dim=-1)


def truncated_signature(increments: Tensor, depth: int = 2) -> Tensor:
    """Exact depth-1/2 signature of a concatenated piecewise-linear path.

    ``increments`` has shape ``[batch, steps, channels]``. The depth-two
    update follows Chen's identity: S2 <- S2 + S1⊗dx + 1/2 dx⊗dx.
    """
    if increments.ndim != 3:
        raise ValueError("increments must have shape [batch, steps, channels]")
    if depth not in {1, 2}:
        raise ValueError("this implementation supports signature depth 1 or 2")
    batch, _, channels = increments.shape
    first = torch.zeros(batch, channels, dtype=increments.dtype, device=increments.device)
    second = None
    if depth == 2:
        second = torch.zeros(
            batch, channels, channels, dtype=increments.dtype, device=increments.device
        )
    for delta in increments.unbind(dim=1):
        if second is not None:
            second = second + torch.einsum("bi,bj->bij", first, delta)
            second = second + 0.5 * torch.einsum("bi,bj->bij", delta, delta)
        first = first + delta
    if second is None:
        return first
    return torch.cat([first, second.flatten(start_dim=1)], dim=-1)


def signature_dimension(channels: int, depth: int) -> int:
    if depth == 1:
        return channels
    if depth == 2:
        return channels + channels * channels
    raise ValueError("this implementation supports signature depth 1 or 2")
