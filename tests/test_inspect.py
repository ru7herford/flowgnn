"""Check recorded message updates and the sensitivity map against independent calculations."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inspect import round_influence, sensitivity, trace_messages
from src.model import FlowGNN


def fixture():
    """A bidirectional path gives an independently known communication distance."""
    torch.manual_seed(7)
    model = FlowGNN(4, 3, 3, 8, 2).double().eval()
    index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    return model, torch.randn(5, 4, dtype=torch.double), index, torch.randn(8, 3, dtype=torch.double)


def test_trace_uses_complete_graph_and_all_incoming_messages():
    """A Python sum and the actual forward prediction must agree with the exported trace."""
    model, x, index, a = fixture()
    trace = trace_messages(model, x, index, a, [1, 2])
    with torch.no_grad():
        assert np.allclose(trace["predicted"], model(x, index, a).numpy(), rtol=0, atol=1e-12)
        for layer in range(2):
            for slot, target in enumerate(trace["targets"]):
                total = trace["messages"][layer, trace["destinations"] == target].sum(axis=0)
                assert np.allclose(total, trace["incoming"][layer, slot], atol=1e-12)
                h = torch.tensor(trace["before"][layer, slot])
                summed = torch.tensor(total)
                update = h + model.layers[layer].node(torch.cat([h, summed]))
                assert np.allclose(update.numpy(), trace["after"][layer, slot], atol=1e-12)


def test_sensitivity_matches_finite_differences_and_reach():
    """Each derivative equals a central difference at several step sizes; nodes beyond reach give zero."""
    model, x, index, a = fixture()  # 2 layers on a path of 5 nodes
    grad = sensitivity(model, x, index, a, node=0)
    with torch.no_grad():
        for step in (1e-3, 1e-4, 1e-5):
            for node, feature in [(0, 0), (1, 2), (2, 3)]:
                up, down = x.clone(), x.clone()
                up[node, feature] += step
                down[node, feature] -= step
                difference = (model(up, index, a)[0, 0] - model(down, index, a)[0, 0]) / (2 * step)
                assert abs(difference.item() - grad[node, feature]) < 1e-6
    assert np.abs(grad[2]).max() > 0 and np.all(grad[3:] == 0)  # two layers reach two links, no further


def test_round_influence_grows_one_link_per_round():
    """After round r the state depends on nodes up to r links away and no further; the values equal the full
    Jacobian of that state, computed independently."""
    model, x, index, a = fixture()  # 2 layers on a path of 5 nodes; node 0 is at one end
    share = round_influence(model, x, index, a, node=0)
    for r in range(2):
        assert np.all(share[r, : r + 2] > 0) and np.all(share[r, r + 2:] == 0)
        assert abs(share[r].sum() - 1) < 1e-12

        def state(inputs, r=r):
            h, e = model.encode_node(inputs), model.encode_edge(a)
            for layer in model.layers[: r + 1]:
                h, e = layer(h, e, index)
            return h[0]

        jacobian = torch.autograd.functional.jacobian(state, x)  # (state numbers, nodes, inputs)
        score = jacobian.abs().sum(dim=(0, 2)).numpy()
        assert np.allclose(share[r], score / score.sum(), rtol=0, atol=1e-12)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
