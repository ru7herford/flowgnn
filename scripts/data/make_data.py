"""Generate the FlowGNN dataset (data/cases/*.npz and data/manifest.json).

For every case in scripts/data/dataset.json:
  1. mesh the channel minus the obstacle twice with gmsh: a fine solve mesh with a
     refined wake box, and the coarser training mesh the network reads,
  2. solve steady incompressible laminar flow on the solve mesh with SU2,
  3. interpolate u, v, p linearly onto the training mesh and store it in data/cases/.

Working files (meshes, SU2 config, log, history, a reference plot) go to lab/runs/data/<case>/.

    pip install -r requirements-data.txt
    python scripts/data/make_data.py [--su2 PATH] [--threads 6] [--jobs 4] [--only CASE_ID ...]

SU2_CFD is found from --su2, then the SU2_CFD environment variable, then PATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gmsh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent  # this script, dataset.json and su2.cfg
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import NODE_LABELS, WAKE_WINDOW  # noqa: E402

WORK = ROOT / "lab" / "runs" / "data"  # meshes, solver logs and plots: large, kept out of the repository
DATA = ROOT / "data"
SU2_VERSION = "8.5.0"
DISTANCE_SAMPLING = 200  # points per obstacle curve used by gmsh's distance field
# SU2 marker -> node label. Where boundaries meet, the later entry wins, so corner nodes
# belong to the inlet/outlet and obstacle nodes are never overwritten.
MARKERS = {"top": "side wall", "bottom": "side wall", "outlet": "outlet", "inlet": "inlet", "obstacle": "obstacle"}


def outline(case: dict, n: int = 128) -> np.ndarray:
    """Obstacle boundary as an (n, 2) counter-clockwise polygon (not closed)."""
    shape = case["shape"]
    if shape == "ellipse":
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        points = np.stack([case["a"] * np.cos(t), case["b"] * np.sin(t)], axis=1)
    elif shape == "rectangle":
        w, h = case["w"] / 2, case["h"] / 2
        points = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    elif shape == "triangle":  # equilateral, centroid at the origin, one vertex up
        t = np.pi / 2 + 2.0 * np.pi * np.arange(3) / 3
        points = case["side"] / np.sqrt(3.0) * np.stack([np.cos(t), np.sin(t)], axis=1)
    elif shape == "naca":
        points = naca_4digit(case["naca"], case["chord"], n // 2)
    else:
        raise ValueError(f"unknown shape '{shape}' (use ellipse, rectangle, triangle or naca)")
    # Airfoils use the aerodynamic convention: a positive angle of attack lifts the nose.
    angle = np.radians(-case["aoa_deg"] if shape == "naca" else case["angle_deg"])
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]]) + [case["cx"], case["cy"]]


def naca_4digit(code: str, chord: float, n: int) -> np.ndarray:
    """NACA 4-digit airfoil, mid-chord at the origin, leading edge on the left.

    Standard thickness and camber-line formulas, with the trailing-edge coefficient
    changed so the trailing edge closes to a point. Cosine spacing puts more points
    near both edges, where the surface curves fastest.
    """
    camber, position, thickness = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n + 1)))
    half = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if camber > 0:
        front = x < position
        line = np.where(front, camber / position**2 * (2 * position * x - x**2),
                        camber / (1 - position) ** 2 * (1 - 2 * position + 2 * position * x - x**2))
        slope = np.where(front, 2 * camber / position**2 * (position - x),
                         2 * camber / (1 - position) ** 2 * (position - x))
    else:
        line, slope = np.zeros_like(x), np.zeros_like(x)
    theta = np.arctan(slope)
    upper = np.stack([x - half * np.sin(theta), line + half * np.cos(theta)], axis=1)
    lower = np.stack([x + half * np.sin(theta), line - half * np.cos(theta)], axis=1)
    # Counter-clockwise from the trailing edge: upper surface forward, lower surface back.
    loop = np.concatenate([upper[::-1][:-1], lower[:-1]])
    return chord * (loop - [0.5, 0.0])


def frontal_reynolds(case: dict, flow: dict) -> float:
    """Reynolds number on the obstacle's height across the flow, U h / nu."""
    height = float(np.ptp(outline(case, 4096)[:, 1]))
    return flow["reynolds"] * height / flow["reference_length"]


def build_mesh(case: dict, domain: dict, size: dict, path: Path, scale: float = 1.0) -> None:
    """Triangulate the channel minus the obstacle and write an SU2 mesh.

    Cells grow linearly from size_wall at the obstacle to size_far at grow_distance. If the
    settings have a wake box, cells inside it (from wake_upstream before the obstacle to
    wake_downstream behind it, wake_side to either side) are at most wake_size. `scale`
    multiplies every size, for the grid-convergence study.
    """
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        channel = occ.addRectangle(domain["xmin"], domain["ymin"], 0,
                                   domain["xmax"] - domain["xmin"], domain["ymax"] - domain["ymin"])
        if case["shape"] == "ellipse":  # exact ellipse rather than a polygon
            a, b = max(case["a"], case["b"]), min(case["a"], case["b"])  # gmsh wants the major radius first
            body = occ.addDisk(case["cx"], case["cy"], 0, a, b)
            turn = case["angle_deg"] + (90 if case["b"] > case["a"] else 0)
            occ.rotate([(2, body)], case["cx"], case["cy"], 0, 0, 0, 1, np.radians(turn))
        else:
            points = [occ.addPoint(x, y, 0) for x, y in outline(case)]
            lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
            body = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        fluid = occ.cut([(2, channel)], [(2, body)])[0][0][1]
        occ.synchronize()

        # Name boundary curves by where they lie; whatever is not on the channel walls is the obstacle.
        def on(lo: float, hi: float, value: float) -> bool:
            return abs(lo - value) < 1e-6 and abs(hi - value) < 1e-6

        groups = {"inlet": [], "outlet": [], "top": [], "bottom": [], "obstacle": []}
        for _, tag in gmsh.model.getBoundary([(2, fluid)], oriented=False):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, tag)
            if on(x0, x1, domain["xmin"]):
                groups["inlet"].append(tag)
            elif on(x0, x1, domain["xmax"]):
                groups["outlet"].append(tag)
            elif on(y0, y1, domain["ymax"]):
                groups["top"].append(tag)
            elif on(y0, y1, domain["ymin"]):
                groups["bottom"].append(tag)
            else:
                groups["obstacle"].append(tag)
        for name, tags in groups.items():
            gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), name)
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid]), "fluid")

        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", groups["obstacle"])
        field.setNumber(distance, "Sampling", DISTANCE_SAMPLING)
        grow = field.add("Threshold")
        field.setNumber(grow, "InField", distance)
        field.setNumber(grow, "SizeMin", scale * size["size_wall"])
        field.setNumber(grow, "SizeMax", scale * size["size_far"])
        field.setNumber(grow, "DistMin", 0.0)
        field.setNumber(grow, "DistMax", size["grow_distance"])
        background = grow
        if "wake_size" in size:
            box = outline(case)
            wake = field.add("Box")
            field.setNumber(wake, "VIn", scale * size["wake_size"])
            field.setNumber(wake, "VOut", scale * size["size_far"])
            field.setNumber(wake, "XMin", box[:, 0].min() - size["wake_upstream"])
            field.setNumber(wake, "XMax", box[:, 0].max() + size["wake_downstream"])
            field.setNumber(wake, "YMin", box[:, 1].min() - size["wake_side"])
            field.setNumber(wake, "YMax", box[:, 1].max() + size["wake_side"])
            field.setNumber(wake, "Thickness", size["wake_blend"])
            background = field.add("Min")
            field.setNumbers(background, "FieldsList", [grow, wake])
        field.setAsBackgroundMesh(background)
        for option in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(f"Mesh.{option}", 0)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def find_su2(given: str | None) -> str:
    """SU2_CFD from --su2, then $SU2_CFD, then PATH; checks the version it reports."""
    for candidate in (given, os.environ.get("SU2_CFD"), shutil.which("SU2_CFD")):
        if candidate and Path(candidate).is_file():
            su2 = str(Path(candidate).resolve())  # SU2 runs inside each case folder
            banner = subprocess.run([su2, "--help"], capture_output=True, text=True).stdout
            found = re.search(r"SU2 v(\S+)", banner)
            if not found or found.group(1) != SU2_VERSION:
                print(f"warning: {su2} reports SU2 {found.group(1) if found else 'unknown'}; "
                      f"the data was made with {SU2_VERSION}")
            return su2
    raise SystemExit(f"SU2_CFD not found. Install SU2 {SU2_VERSION} (https://su2code.github.io), put SU2_CFD "
                     "on PATH, set SU2_CFD=/path/to/SU2_CFD, or pass --su2 /path/to/SU2_CFD.")


def solve(case_dir: Path, cfg: dict, su2: str, threads: int) -> dict:
    """Run SU2 (one process, `threads` OpenMP threads) in case_dir; summarise its history file."""
    flow = cfg["flow"]
    template = (HERE / "su2.cfg").read_text()
    (case_dir / "case.cfg").write_text(template.format(
        density=flow["density"],
        velocity=flow["velocity"],
        viscosity=flow["density"] * flow["velocity"] * flow["reference_length"] / flow["reynolds"],
        reference_length=flow["reference_length"],
        **cfg["convergence"],
    ))
    start = time.perf_counter()
    with open(case_dir / "solver.log", "w") as log:
        process = subprocess.run([su2, "-t", str(threads), "case.cfg"], cwd=case_dir,
                                 stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start

    history = case_dir / "history.csv"
    rows = list(csv.reader(open(history))) if history.exists() else []
    if process.returncode != 0 or len(rows) < 2:  # a crash can leave a header-only history
        return {"converged": False, "seconds": round(seconds, 1)}
    last = {name.strip().strip('"'): float(value) for name, value in zip(rows[0], rows[-1])}
    return {
        "converged": last["rms[P]"] <= cfg["convergence"]["rms_pressure_log10"],
        "iterations": int(last["Inner_Iter"]) + 1,
        "final_rms_pressure_log10": round(last["rms[P]"], 3),
        "final_rms_velocity_log10": round(max(last["rms[U]"], last["rms[V]"]), 3),
        "drag_coefficient": round(last["CD"], 5),
        "lift_coefficient": round(last["CL"], 5),
        "seconds": round(seconds, 1),
    }


def read_su2_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Node coordinates, triangles and boundary edges per marker of an SU2 mesh."""
    lines = path.read_text().splitlines()
    points = triangles = None
    markers: dict[str, np.ndarray] = {}
    name = None
    i = 0
    while i < len(lines):
        key, _, value = lines[i].partition("=")
        key = key.strip()
        if key == "NELEM":
            n = int(value)
            rows = np.array([line.split()[:4] for line in lines[i + 1:i + 1 + n]], dtype=np.int64)
            if not (rows[:, 0] == 5).all():  # SU2 element type 5 = triangle
                raise ValueError(f"{path}: expected only triangles")
            triangles = rows[:, 1:4]
            i += n
        elif key == "NPOIN":
            n = int(value.split()[0])
            points = np.array([line.split()[:2] for line in lines[i + 1:i + 1 + n]], dtype=np.float64)
            i += n
        elif key == "MARKER_TAG":
            name = value.strip()
        elif key == "MARKER_ELEMS":
            n = int(value)
            markers[name] = np.array([line.split()[1:3] for line in lines[i + 1:i + 1 + n]], dtype=np.int64)
            i += n
        i += 1
    return points, triangles, markers


def node_labels(markers: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Boundary type of every node from the SU2 boundary markers (fluid where no marker applies)."""
    node_type = np.full(n, NODE_LABELS["fluid"], dtype=np.int8)
    for marker, label in MARKERS.items():
        node_type[np.unique(markers[marker])] = NODE_LABELS[label]
    return node_type


def read_solution(case_dir: Path) -> dict:
    """Solve mesh and its u, v, p (SU2 writes flow.vtu nodes in mesh.su2 order, single precision)."""
    pos, tri, markers = read_su2_mesh(case_dir / "solve.su2")
    flow = pv.read(case_dir / "flow.vtu")
    if np.abs(np.asarray(flow.points)[:, :2] - pos).max() > 1e-4:
        raise ValueError(f"{case_dir}: flow.vtu nodes do not match solve.su2")
    fields = np.column_stack([np.asarray(flow.point_data["Velocity"])[:, :2], np.asarray(flow.point_data["Pressure"])])
    return {"pos": pos, "tri": tri, "node_type": node_labels(markers, len(pos)), "fields": fields.astype(np.float64)}


def interpolate(solution: dict, points: np.ndarray) -> tuple[np.ndarray, int]:
    """Linear interpolation of u, v, p from the solve mesh to `points`.

    Obstacle nodes on a straight wall lie exactly on the solve mesh's boundary edges, where the
    interpolator can count them as outside; such points take the value of the nearest solve node
    (their velocity is then set to zero anyway). Returns the values and how many points needed that.
    """
    triangulation = Triangulation(solution["pos"][:, 0], solution["pos"][:, 1], solution["tri"])
    values = np.empty((len(points), 3))
    outside = np.zeros(len(points), dtype=bool)
    for k in range(3):
        v = LinearTriInterpolator(triangulation, solution["fields"][:, k])(points[:, 0], points[:, 1])
        outside |= np.ma.getmaskarray(v)
        values[:, k] = np.ma.filled(v, np.nan)
    if outside.any():
        _, nearest = cKDTree(solution["pos"]).query(points[outside])
        values[outside] = solution["fields"][nearest]
    return values, int(outside.sum())


def plot_case(arrays: dict, title: str, path: Path) -> None:
    """Reference streamwise velocity and pressure in the near wake."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    x, y = arrays["pos"].T
    x0, y0, x1, y1 = WAKE_WINDOW
    for ax, values, label in zip(axes, (arrays["U"][:, 0], arrays["p"]), ("u [U]", "p [rho U^2]")):
        image = ax.tripcolor(x, y, arrays["tri"], values, shading="gouraud", cmap="RdBu_r")
        ax.fill(*arrays["outline"].T, color="0.3")
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
        fig.colorbar(image, ax=ax, label=label, shrink=0.8)
    axes[0].set_title(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Mesh, solve and store every case in the settings, then write data/manifest.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "dataset.json")
    parser.add_argument("--su2", help="path to SU2_CFD")
    parser.add_argument("--threads", type=int, default=6, help="OpenMP threads per SU2 run")
    parser.add_argument("--jobs", type=int, default=4, help="SU2 runs at the same time")
    parser.add_argument("--only", nargs="+", metavar="CASE_ID", help="regenerate only these cases")
    args = parser.parse_args()

    su2 = find_su2(args.su2)
    cfg = json.loads(args.config.read_text())
    unknown = set(args.only or []) - {c["id"] for c in cfg["cases"]}
    if unknown:
        raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")
    too_fast = [(c["id"], round(frontal_reynolds(c, cfg["flow"]), 1)) for c in cfg["cases"]
                if frontal_reynolds(c, cfg["flow"]) > cfg["max_frontal_reynolds"]]
    if too_fast:
        raise SystemExit(f"Reynolds number on frontal height above {cfg['max_frontal_reynolds']}: {too_fast}")
    cases = [c for c in cfg["cases"] if not args.only or c["id"] in args.only]
    (DATA / "cases").mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if args.only and manifest_path.exists() else {}
    # Regenerating some cases is only valid if every shared setting is unchanged.
    shared = {k: v for k, v in cfg.items() if k != "cases"}
    if args.only and {k: v for k, v in previous.get("config", {}).items() if k != "cases"} != shared:
        raise SystemExit("settings differ from data/manifest.json; regenerate every case (run without --only)")
    records = {r["id"]: r for r in previous.get("cases", [])}

    print(f"Meshing {len(cases)} cases.", flush=True)
    for case in cases:
        case_dir = WORK / case["id"]
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True)
        build_mesh(case, cfg["domain"], cfg["solve_mesh"], case_dir / "solve.su2")
        build_mesh(case, cfg["domain"], cfg["training_mesh"], case_dir / "training.su2")

    print(f"Solving with SU2, {args.jobs} runs at a time, {args.threads} threads each.", flush=True)
    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(solve, WORK / c["id"], cfg, su2, args.threads): c["id"] for c in cases}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = r = future.result()
            print(f"[{done}/{len(cases)}] {futures[future]}: " + (
                f"{r['iterations']} iterations, {r['seconds']} s, Cd {r['drag_coefficient']:.4f}" if r["converged"]
                else f"not converged ({r})"), flush=True)

    for case in cases:
        case_dir, result = WORK / case["id"], results[case["id"]]
        npz = DATA / "cases" / f"{case['id']}.npz"
        record = {"id": case["id"], "split": case["split"],
                  "frontal_reynolds": round(frontal_reynolds(case, cfg["flow"]), 2), **result}
        if result["converged"]:
            solution = read_solution(case_dir)
            pos, tri, markers = read_su2_mesh(case_dir / "training.su2")
            node_type = node_labels(markers, len(pos))
            values, outside = interpolate(solution, pos)
            values[node_type == NODE_LABELS["obstacle"], :2] = 0.0  # no slip, exactly
            arrays = {"pos": pos, "tri": tri.astype(np.int32), "node_type": node_type, "outline": outline(case),
                      "U": values[:, :2].astype(np.float32), "p": values[:, 2].astype(np.float32),
                      "params": np.array(json.dumps(case))}
            np.savez_compressed(npz, **arrays)
            plot_case(arrays, case["id"], case_dir / "reference.png")
            record.update(solve_mesh_nodes=len(solution["pos"]), nodes=len(pos), triangles=len(tri),
                          nearest_node_fallbacks=outside)
        else:
            npz.unlink(missing_ok=True)
            print(f"{case['id']}: left out of the dataset")
        records[case["id"]] = record

    version = re.search(r"Release (\S+)", (WORK / cases[-1]["id"] / "solver.log").read_text())
    manifest = {
        "solver": f"SU2 {version.group(1) if version else 'unknown'} (OpenMP build, one process, {args.threads} threads)",
        "mesher": f"gmsh {gmsh.__version__}",
        "config": cfg,
        "cases": [records[c["id"]] for c in cfg["cases"] if c["id"] in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    kept = sum(r["converged"] for r in manifest["cases"])
    print(f"Done: {kept}/{len(manifest['cases'])} cases in data/cases/, summary in data/manifest.json")


if __name__ == "__main__":
    main()
