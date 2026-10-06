"""Record real message updates and Jacobians from a saved graph network.

States and messages are internal vectors. Their components are not flow variables.
Inference always uses the full graph; target IDs only select what is recorded.
"""

from pathlib import Path

import numpy as np
import torch

from src.graph import Case, Normaliser, edge_inputs, node_inputs
from src.model import FlowGNN


def load_checkpoint(path: Path, case: Case, device: str = "cpu") -> tuple:
    """Load weights and the original training scales, then build the complete case graph."""
    saved = torch.load(path, map_location=device, weights_only=True)
    norms = {k: Normaliser.from_state(v) for k, v in saved["normalisation"].items()}
    x = norms["x"](node_inputs(case, saved["input_set"]))
    a = norms["edge"](edge_inputs(case))
    model = FlowGNN(x.shape[1], a.shape[1], 3, **saved["model"]).to(device).eval()
    model.load_state_dict(saved["weights"])
    node_values = torch.as_tensor(x, device=device)
    connections = torch.as_tensor(case.edge_index, device=device).long()
    edge_values = torch.as_tensor(a, device=device)
    return model, node_values, connections, edge_values, norms


@torch.no_grad()
def trace_messages(model, x, edge_index, edge_attr, targets: list[int]) -> dict:
    """Replay the existing layers and retain all incoming messages for selected nodes."""
    if (not targets or len(set(targets)) != len(targets)
            or min(targets) < 0 or max(targets) >= len(x)):
        raise ValueError("Targets must be distinct node IDs in the full graph")
    target = torch.as_tensor(targets, device=x.device)
    selected = torch.isin(edge_index[1], target)
    h, e = model.encode_node(x), model.encode_edge(edge_attr)
    before, incoming, after, messages = [], [], [], []
    for layer in model.layers:
        before.append(h[target].cpu().numpy())
        h, e = layer(h, e, edge_index)
        summed = torch.zeros_like(h).index_add_(0, edge_index[1], e.to(h.dtype))
        incoming.append(summed[target].cpu().numpy())
        after.append(h[target].cpu().numpy())
        messages.append(e[selected].cpu().numpy())
    return {
        "targets": targets,
        "sources": edge_index[0, selected].cpu().numpy(),
        "destinations": edge_index[1, selected].cpu().numpy(),
        "before": np.stack(before),
        "incoming": np.stack(incoming),
        "after": np.stack(after),
        "messages": np.stack(messages),
        "predicted": model.decode(h).cpu().numpy(),
    }


def sensitivity(model, x, edge_index, edge_attr, node: int, output: int = 0) -> np.ndarray:
    """One row of the network's Jacobian: d(prediction at `node`)/d(inputs of every node).

    Returns an (N, F) array: the derivative of output `output` (0 = u, horizontal velocity) at `node`
    with respect to every input of every node, in the normalised units the network reads. Backpropagation
    computes it in one backward pass (reverse-mode differentiation, the adjoint method). It is local
    sensitivity at the actual inputs. Nodes outside the receptive field (more links away than the network
    has layers) get exactly zero.
    """
    x = x.detach().clone().requires_grad_(True)
    model(x, edge_index, edge_attr)[node, output].backward()
    return x.grad.cpu().numpy()


def round_influence(model, x, edge_index, edge_attr, node: int) -> np.ndarray:
    """Influence distribution of `node`'s state after each round (Xu et al. 2018, Definition 3.1).

    For round r, the influence of node j is the sum of |d h_r[node, k] / d x[j, f]| over the 64 state numbers k
    and the inputs f of node j, divided by the sum over all nodes. Returns a (rounds, N) array. Row r is exactly
    zero for nodes more than r + 1 links away: the receptive field after r + 1 rounds. One backward pass per
    state number and round (12 x 64 here).
    """
    x = x.detach().clone().requires_grad_(True)
    h, e = model.encode_node(x), model.encode_edge(edge_attr)
    rows = []
    for layer in model.layers:
        h, e = layer(h, e, edge_index)
        score = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        for k in range(h.shape[1]):
            (grad,) = torch.autograd.grad(h[node, k], x, retain_graph=True)
            score += grad.abs().sum(dim=1)
        rows.append((score / score.sum()).detach().cpu().numpy())
    return np.stack(rows)
