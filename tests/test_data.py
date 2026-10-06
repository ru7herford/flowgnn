"""Checks of the reference data. Run with `python tests/test_data.py` (or pytest)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import OBSTACLE, Case  # noqa: E402

DATA = ROOT / "data"
CASES = sorted((DATA / "cases").glob("*.npz"))
VALIDATION = json.loads((DATA / "validation.json").read_text())


def interpolate(case: Case, points: np.ndarray, column: int) -> np.ndarray:
    """Linear interpolation of one target column at points, inside the mesh triangles (NaN outside)."""
    a, b, c = (case.pos[case.tri[:, k]] for k in range(3))
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    values = np.full(len(points), np.nan)
    for i, (x, y) in enumerate(points):
        l1 = ((b[:, 0] - x) * (c[:, 1] - y) - (c[:, 0] - x) * (b[:, 1] - y)) / det
        l2 = ((c[:, 0] - x) * (a[:, 1] - y) - (a[:, 0] - x) * (c[:, 1] - y)) / det
        l3 = 1 - l1 - l2
        hit = np.nonzero((l1 >= -1e-12) & (l2 >= -1e-12) & (l3 >= -1e-12))[0]
        if len(hit):
            t = hit[0]
            values[i] = (l1[t] * case.target[case.tri[t, 0], column] + l2[t] * case.target[case.tri[t, 1], column]
                         + l3[t] * case.target[case.tri[t, 2], column])
    return values


def test_frontal_reynolds_number_stays_within_the_reference():
    """Check that the reference Reynolds number bounds every obstacle's frontal value."""
    # Re on frontal height, U h / nu with U = 1 and nu = 1 / 40, stays at or below 40 (h <= 1).
    for path in CASES:
        case = Case.load(path)
        assert np.ptp(case.outline[:, 1]) <= 1.0 + 1e-9, path.name


def test_no_slip_on_the_obstacle_and_inflow_far_upstream():
    """Stored fields must satisfy the obstacle and upstream boundary values."""
    for path in CASES:
        case = Case.load(path)
        assert np.all(case.target[case.node_type == OBSTACLE, :2] == 0), path.name
        upstream = case.pos[:, 0] < -15
        assert np.allclose(case.target[upstream, :2], [1.0, 0.0], atol=0.02), path.name


def test_every_case_converged():
    """Require every recorded solver run to meet the saved convergence threshold."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert len(manifest["cases"]) == len(CASES) == 50
    target = manifest["config"]["convergence"]["rms_pressure_log10"]
    assert all(c["converged"] and c["final_rms_pressure_log10"] <= target for c in manifest["cases"])


def test_circle_passes_its_acceptance_bands():
    """The circle solve lies inside the bands set before the runs, and its grid error is small."""
    # The setup check (scripts/data/validate_circle.py), against bands set before the runs.
    accept, dataset = VALIDATION["acceptance"], next(m for m in VALIDATION["meshes"] if m["mesh"] == "dataset")
    for quantity in ("drag_coefficient", "wake_length", "separation_angle"):
        low, high = accept[quantity]
        assert low <= dataset[quantity] <= high, quantity
        assert VALIDATION["grid_convergence"][quantity]["gci_dataset_percent"] < accept["gci_dataset_percent"], quantity


def test_stored_circle_keeps_the_solved_wake():
    """Check that sampling onto the training mesh preserves the solved wake length."""
    # The training mesh stores the solve-mesh values at its nodes. Where the bubble ends its edges are about 0.3
    # long, so reading the end by straight lines between nodes can be off by half an edge, but no more; solving
    # on the training mesh itself puts the end 0.43 short (data/validation.json).
    case = Case.load(DATA / "cases" / "ellipse_01.npz")
    xs = np.linspace(0.55, 4.0, 691)
    u = interpolate(case, np.column_stack([xs, np.zeros_like(xs)]), 0)
    k = np.nonzero((u[:-1] < 0) & (u[1:] >= 0))[0][0]
    end = xs[k] - u[k] * (xs[k + 1] - xs[k]) / (u[k + 1] - u[k])
    solved = next(m for m in VALIDATION["meshes"] if m["mesh"] == "dataset")["wake_length"]
    assert abs((end - 0.5) - solved) < 0.17, (end - 0.5, solved)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
