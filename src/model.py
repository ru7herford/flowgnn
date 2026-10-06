"""Encode-process-decode graph network (as in MeshGraphNets), in plain PyTorch.

Each processor layer updates every edge from its two end nodes, sums the
edges arriving at each node, and updates the node. After L layers a node has
information from nodes up to L edges away, and from no node further away.

Names used below: h is a node state, e is an edge state, src/dst are the
source/destination node IDs. An mlp is a small two-layer learned rule.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs: int, hidden: int, outputs: int, norm: bool = True) -> nn.Sequential:
    """Build the small learned rule used by encoders, updates and the decoder."""
    layers = [nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs)]
    if norm:
        layers.append(nn.LayerNorm(outputs))
    return nn.Sequential(*layers)


class Layer(nn.Module):
    """One round of message passing, with residual updates of edges and nodes."""

    def __init__(self, hidden: int):
        """Create one edge-update rule and one node-update rule."""
        super().__init__()
        self.edge = mlp(3 * hidden, hidden, hidden)
        self.node = mlp(2 * hidden, hidden, hidden)

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update edges, sum incoming messages, then update every node."""
        src, dst = edge_index[0], edge_index[1]
        # The same as self.edge(torch.cat([h[src], h[dst], e], dim=1)), but the first linear layer is
        # applied to the N nodes before gathering to the E (about 6N) edges, so the widest matmul and
        # the concatenation are never formed.
        first, rest = self.edge[0], self.edge[1:]
        w_src, w_dst, w_edge = first.weight.split(h.shape[1], dim=1)
        z = (F.linear(h, w_src).index_select(0, src)
             + F.linear(h, w_dst).index_select(0, dst)
             + F.linear(e, w_edge, first.bias))
        e = e + rest(z)
        incoming = torch.zeros_like(h).index_add_(0, dst, e.to(h.dtype))  # sum over neighbours
        h = h + self.node(torch.cat([h, incoming], dim=1))
        return h, e


class FlowGNN(nn.Module):
    """Turn mesh inputs into flow predictions through repeated local updates."""

    def __init__(self, node_inputs: int, edge_inputs: int, outputs: int, hidden: int, layers: int):
        """Build encoders, communication rounds and the final flow decoder."""
        super().__init__()
        self.encode_node = mlp(node_inputs, hidden, hidden)
        self.encode_edge = mlp(edge_inputs, hidden, hidden)
        self.layers = nn.ModuleList(Layer(hidden) for _ in range(layers))
        self.decode = mlp(hidden, hidden, outputs, norm=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode input features, exchange messages and decode flow variables."""
        h, e = self.encode_node(x), self.encode_edge(edge_attr)
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
        return self.decode(h)
