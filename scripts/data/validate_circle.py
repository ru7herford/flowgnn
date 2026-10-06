"""Check the SU2 setup on the circle at Re 40 and estimate its grid error (writes data/validation.json).

The circle case of the dataset (ellipse_01, diameter 1) is solved with the same settings on
three solve meshes, every cell size 2x, 1x (the dataset's solve mesh) and 0.5x, and once
directly on the training mesh. Drag, wake length and separation angle are compared with
Dennis & Chang (1970) and Gautier, Biau & Lamballais (2013); the grid convergence index
(Celik et al. 2008) gives the discretisation error of the dataset's solve mesh.

    python scripts/data/validate_circle.py [--su2 PATH] [--threads 6] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation

import make_data as md

OUT = md.ROOT / "lab" / "runs" / "circle"  # solver working files, kept out of the repository
LEVELS = {"coarse": 2.0, "dataset": 1.0, "fine": 0.5}  # cell-size multiplier of the solve mesh
PROBE = 0.002  # separation is read from the tangential velocity this far from the wall (inside the first cell)
REFERENCES = [
    {"name": "Dennis & Chang (1970)", "method": "finite differences, steady equations",
     "drag_coefficient": 1.522, "wake_length": 2.345, "separation_angle": 126.2,
     "link": "https://doi.org/10.1017/S0022112070001428"},
    {"name": "Gautier, Biau & Lamballais (2013)", "method": "spectral, unbounded domain (fine grid, r = 40 D)",
     "drag_coefficient": 1.4931, "wake_length": 2.2360, "separation_angle": 126.39,
     "link": "https://arxiv.org/abs/1310.6641"},
]
# Set before the runs: drag within 3% of 1.52 (side-wall blockage adds 2-3%), wake length inside the spread of
# the published values, separation angle 125-128 degrees (the published spread is 124.4-127.3, so this band ends
# 0.7 degrees above it), and a grid error below 2% on the dataset's mesh.
ACCEPTANCE = {"drag_coefficient": [1.49, 1.57], "wake_length": [2.15, 2.35], "separation_angle": [125.0, 128.0],
              "gci_dataset_percent": 2.0}
# Spread of the 17 published values Gautier et al. collect in their Table 1.
LITERATURE_RANGE = {"drag_coefficient": [1.48, 1.62], "wake_length": [2.13, 2.35], "separation_angle": [124.4, 127.3]}


def measure(case: dict, case_dir: Path) -> dict:
    """Wake length (from the rear of the circle, in diameters) and separation angle (from the front, degrees)."""
    sol = md.read_solution(case_dir)
    tri = Triangulation(sol["pos"][:, 0], sol["pos"][:, 1], sol["tri"])
    u = LinearTriInterpolator(tri, sol["fields"][:, 0])
    v = LinearTriInterpolator(tri, sol["fields"][:, 1])
    cx, cy, radius = case["cx"], case["cy"], case["a"]
    diameter = 2 * radius

    xs = np.linspace(cx + radius + 1e-4, cx + radius + 8, 16001)
    us = np.asarray(u(xs, np.full_like(xs, cy)))
    k = np.nonzero((us[:-1] < 0) & (us[1:] >= 0))[0][0]
    x_end = xs[k] - us[k] * (xs[k + 1] - xs[k]) / (us[k + 1] - us[k])

    angles = []
    phi = np.radians(np.linspace(0.05, 179.95, 36000))  # from the rear (phi = 0) towards the front
    for side in (1, -1):
        x = cx + (radius + PROBE) * np.cos(side * phi)
        y = cy + (radius + PROBE) * np.sin(side * phi)
        # velocity along the surface, positive from the front towards the rear
        tangential = np.asarray(u(x, y)) * np.sin(phi) - side * np.asarray(v(x, y)) * np.cos(phi)
        # attached flow runs rearwards (positive); inside the bubble it runs forwards
        j = np.nonzero((tangential[:-1] < 0) & (tangential[1:] >= 0))[0][0]
        rear = phi[j] - tangential[j] * (phi[j + 1] - phi[j]) / (tangential[j + 1] - tangential[j])
        angles.append(180.0 - math.degrees(rear))
    wall = int((sol["node_type"] == md.NODE_LABELS["obstacle"]).sum())
    return {"nodes": len(sol["pos"]), "wall_nodes": wall, "wake_length": round(float(x_end - cx - radius) / diameter, 4),
            "separation_angle": round(float(np.mean(angles)), 3), "separation_angle_upper_lower": [round(a, 3) for a in angles]}


def gci(fine: float, medium: float, coarse: float, n_fine: int, n_medium: int, n_coarse: int) -> dict:
    """Grid convergence index, Celik et al. (2008), with h = (1 / N)^(1/2) in 2D."""
    r21, r32 = math.sqrt(n_fine / n_medium), math.sqrt(n_medium / n_coarse)
    e21, e32 = medium - fine, coarse - medium
    s = math.copysign(1.0, e32 / e21)
    p = 2.0
    for _ in range(200):
        q = math.log((r21**p - s) / (r32**p - s))
        p = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
    extrapolated = (r21**p * fine - medium) / (r21**p - 1)
    ea21 = abs((fine - medium) / fine)
    return {"observed_order": round(p, 2), "monotonic": s > 0, "extrapolated": round(float(extrapolated), 4),
            "gci_fine_percent": round(100 * 1.25 * ea21 / (r21**p - 1), 2),
            # the uncertainty of the medium grid, which is the dataset's solve mesh
            "gci_dataset_percent": round(100 * 1.25 * ea21 * r21**p / (r21**p - 1), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=md.HERE / "dataset.json")
    parser.add_argument("--su2", help="path to SU2_CFD")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    su2 = md.find_su2(args.su2)
    cfg = json.loads(args.config.read_text())
    case = next(c for c in cfg["cases"] if c["id"] == "ellipse_01")
    assert case["shape"] == "ellipse" and case["a"] == case["b"], "ellipse_01 must be the circle"

    runs = {}
    for name, scale in LEVELS.items():
        runs[name] = OUT / name
        shutil.rmtree(runs[name], ignore_errors=True)
        runs[name].mkdir(parents=True)
        md.build_mesh(case, cfg["domain"], cfg["solve_mesh"], runs[name] / "solve.su2", scale)
    runs["training mesh"] = OUT / "training"
    shutil.rmtree(runs["training mesh"], ignore_errors=True)
    runs["training mesh"].mkdir(parents=True)
    md.build_mesh(case, cfg["domain"], cfg["training_mesh"], runs["training mesh"] / "solve.su2")

    with ThreadPoolExecutor(args.jobs) as pool:
        solved = dict(zip(runs, pool.map(lambda d: md.solve(d, cfg, su2, args.threads), runs.values())))
    rows = {}
    for name, directory in runs.items():
        r = solved[name]
        if not r["converged"]:
            raise SystemExit(f"{name}: SU2 did not converge ({r})")
        rows[name] = {"mesh": name, "cell_size_multiplier": LEVELS.get(name), **measure(case, directory),
                      "drag_coefficient": r["drag_coefficient"], "lift_coefficient": r["lift_coefficient"],
                      "iterations": r["iterations"], "final_rms_pressure_log10": r["final_rms_pressure_log10"],
                      "seconds": r["seconds"]}
        print(f"{name:14s} {rows[name]['nodes']:7d} nodes  Cd {r['drag_coefficient']:.4f}  "
              f"Lw/D {rows[name]['wake_length']:.4f}  theta_s {rows[name]['separation_angle']:.2f}  "
              f"({r['iterations']} iterations, {r['seconds']} s)", flush=True)

    f, m, c = rows["fine"], rows["dataset"], rows["coarse"]
    convergence = {q: gci(f[q], m[q], c[q], f["nodes"], m["nodes"], c["nodes"])
                   for q in ("drag_coefficient", "wake_length", "separation_angle")}
    for q, g in convergence.items():
        print(f"GCI {q}: {g}")
    blockage = 2 * case["a"] / (cfg["domain"]["ymax"] - cfg["domain"]["ymin"])
    result = {
        "check": "circle (ellipse_01), diameter D = 1, Re = U D / nu = 40, dataset settings",
        "quantities": {
            "drag_coefficient": "Cd = drag / (0.5 rho U^2 D), from SU2",
            "wake_length": "Lw / D: centreline distance from the back of the circle to where u changes sign",
            "separation_angle": f"degrees from the front stagnation point to where the velocity along the wall "
                                f"changes sign, read {PROBE} D from the wall, mean of upper and lower side",
        },
        "domain": {**cfg["domain"], "blockage": blockage},
        "meshes": list(rows.values()),
        "grid_convergence": convergence,
        "references": REFERENCES,
        "literature_range": LITERATURE_RANGE,
        "acceptance": ACCEPTANCE,
        "su2": md.SU2_VERSION,
    }
    (md.DATA / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote data/validation.json")


if __name__ == "__main__":
    main()
