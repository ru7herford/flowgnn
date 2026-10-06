"""Checks of the network, error metric and validation checkpoint selection. Run with `python tests/test_model.py` (or pytest)."""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowgnn import errors, snapshot_epochs, train_one  # noqa: E402
from src.graph import INPUT_SETS, Case, build_graphs, edges, node_inputs  # noqa: E402
from src.model import FlowGNN  # noqa: E402


def test_edge_update_is_the_concatenated_mlp():
    """Compare the faster edge calculation with the direct concatenated learned rule."""
    # The layer applies the edge network's first linear map to nodes before gathering them to edges;
    # that must equal the textbook form phi_e([h_i, h_j, e_ij]).
    torch.manual_seed(0)
    model = FlowGNN(7, 3, 3, hidden=16, layers=1).double()
    layer = model.layers[0]
    h, e = torch.randn(5, 16, dtype=torch.double), torch.randn(10, 16, dtype=torch.double)
    edge_index = torch.as_tensor(edges(np.array([[0, 1, 2], [1, 3, 2], [3, 4, 2]])))[:, :10]
    src, dst = edge_index
    h_new, e_new = layer(h, e, edge_index)
    e_ref = e + layer.edge(torch.cat([h[src], h[dst], e], dim=1))
    h_ref = h + layer.node(torch.cat([h, torch.zeros_like(h).index_add_(0, dst, e_ref)], dim=1))
    assert torch.allclose(e_new, e_ref) and torch.allclose(h_new, h_ref)


def test_network_hears_only_as_far_as_it_has_layers():
    """Perturb an input outside the communication neighbourhood and check its effect."""
    # A path graph 0-1-2-3-4-5: with 2 layers, changing node 0's input must not move node 3.
    path = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    torch.manual_seed(0)
    model = FlowGNN(4, 3, 3, hidden=8, layers=2).double().eval()
    x, a = torch.randn(6, 4, dtype=torch.double), torch.randn(10, 3, dtype=torch.double)
    moved = x.clone()
    moved[0] += 1.0
    with torch.no_grad():
        change = (model(moved, path, a) - model(x, path, a)).abs().sum(dim=1)
    assert change[:3].min() > 0 and torch.all(change[3:] == 0)


def test_error_metric_is_relative_to_the_disturbance():
    """Exact flow scores zero; predicting undisturbed inflow scores one."""
    case = Case.load(ROOT / "data" / "cases" / "triangle_05.npz")
    exact = errors(case, case.target.copy())
    assert exact["all"] == {"velocity": 0.0, "pressure": 0.0}
    # Predicting the undisturbed inflow everywhere (u = 1, v = 0, p = 0) scores exactly 1 on both.
    free = np.zeros_like(case.target)
    free[:, 0] = 1.0
    assert np.allclose(list(errors(case, free)["all"].values()), [1.0, 1.0])


def test_input_sets_add_columns_in_order():
    """Input widths and shared columns must agree between the four models."""
    case = Case.load(ROOT / "data" / "cases" / "triangle_05.npz")
    widths = {s: node_inputs(case, s).shape[1] for s in INPUT_SETS}
    assert widths == {"M0": 7, "M1": 10, "M2": 13, "M3": 16}
    assert np.array_equal(node_inputs(case, "M3")[:, :10], node_inputs(case, "M1"))
    assert np.array_equal(node_inputs(case, "M3")[:, 10:], node_inputs(case, "M2")[:, 7:])


def test_normalisation_uses_training_cases_only():
    """Held-out fields must not contribute to fitted normalisation statistics."""
    cases = [Case.load(ROOT / "data" / "cases" / f"{n}.npz") for n in ("ellipse_01", "triangle_05")]
    graphs, norms = build_graphs(cases, "M1", "cpu")
    train_target = cases[0].target if cases[0].split == "train" else cases[1].target
    assert np.allclose(norms["y"].mean, train_target.mean(axis=0), atol=1e-5)
    assert np.allclose(norms["x"].mean[:7], 0) and np.allclose(norms["x"].std[:7], 1)


def test_snapshot_epochs():
    """Saved epochs must include the early updates and the final training epoch."""
    epochs = snapshot_epochs(5, 150)
    assert epochs[:4] == [0, 1, 2, 5] and epochs[-1] == 150 and len(epochs) == 33


def test_validation_selection_replays_and_ignores_test_targets():
    """Saved weights reproduce the best validation score; changed test targets cannot choose them."""
    pos = np.array([[-1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    target = np.array([[.3, .2, 1.], [.7, -.2, -.5], [1.2, .4, .5]], dtype=np.float32)
    cases = [Case(split, split, {"shape": "triangle"}, pos, np.array([[0, 1, 2]]),
                  np.array([0, 1, 4]), pos, target * factor)
             for split, factor in [("train", 1), ("val", -1), ("test", 1)]]
    cfg = {"model": {"hidden": 8, "layers": 1},
           "training": {"epochs": 8, "batch_cases": 1, "learning_rate": .02,
                        "gradient_clip": 1., "validate_every": 1},
           "evaluation": {"snapshot_case": "test", "snapshot_every": 4}}
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        result = train_one(cases, "M0", 7, cfg, out / "first", "cpu")
        rows = list(csv.DictReader((out / "first/history.csv").read_text().splitlines()))
        best = min(rows, key=lambda row: float(row["val_mse"]))
        assert result["selected_epoch"] == int(best["epoch"])
        saved = torch.load(out / "first/model.pt", weights_only=True)
        graphs, _ = build_graphs(cases, "M0", "cpu")
        model = FlowGNN(7, 3, 3, hidden=8, layers=1).eval()
        model.load_state_dict(saved["weights"])
        val = next(g for g in graphs if g.case.split == "val")
        with torch.no_grad():
            score = torch.mean((model(val.x, val.edge_index, val.edge_attr) - val.y) ** 2).item()
        assert np.isclose(score, float(best["val_mse"]), atol=1e-6, rtol=0)
        cases[-1].target = cases[-1].target * 100
        changed = train_one(cases, "M0", 7, cfg, out / "changed", "cpu")
        other = torch.load(out / "changed/model.pt", weights_only=True)
        assert changed["selected_epoch"] == result["selected_epoch"]
        assert all(torch.equal(value, other["weights"][key]) for key, value in saved["weights"].items())


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
