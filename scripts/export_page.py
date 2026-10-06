"""Export saved predictions, message traces and training snapshots for the static page.

summary.json records errors, colour bounds and the training replay. Case binaries store float32 positions
and fields, uint32 triangles and stable full-graph node IDs. Progress binaries hold the replay's recorded
horizontal velocity as uint16 over summary progress.range, one snapshot after another; the replay comes from
lab/runs/watch, written by python flowgnn.py --config examples/watch.json. graph.json stores the complete
mesh of the page's triangle, the recorded messages of M3 around one point and that point's inputs;
reach.json stores the complete mesh of one test airfoil with each network's influence distribution at one
wake point; summary.json adds the mean link distance of the influence distribution over several points
and seeds, and the prediction timing from lab/runs/timing.json. Format: docs/data/README.md.

Run after training: python scripts/export_page.py [run_name].
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import INPUT_SETS, NODE_LABELS, OBSTACLE, WAKE_WINDOW, Case, nearest_on_outline, shape_descriptors  # noqa: E402
from src.inspect import load_checkpoint, round_influence, sensitivity, trace_messages  # noqa: E402

HOP_BINS = [0, 3, 6, 9, 12, 15, 20, 25, 30, 40, 50, 64]  # links from the obstacle; points further out are rare
REACH_CASE = "naca3415_aoa8"  # the How far and Influence slides: a test airfoil whose wake point is 19 links away


def hops_from(case: Case, starts) -> np.ndarray:
    """Number of mesh edges between each node and the nearest of `starts` (breadth-first search)."""
    src, dst = case.edge_index
    neighbours = [[] for _ in range(len(case.pos))]
    for a, b in zip(src, dst):
        neighbours[a].append(b)
    hops = np.full(len(case.pos), -1)
    queue = deque(starts)
    hops[list(queue)] = 0
    while queue:
        a = queue.popleft()
        for b in neighbours[a]:
            if hops[b] < 0:
                hops[b] = hops[a] + 1
                queue.append(b)
    return hops


def export_graph(case: Case, folder, shown: dict, out: Path) -> None:
    """Export the complete mesh, one point's recorded messages of M3, that point's inputs, and the influence
    distribution of its state after each round, for the points within reach of the last round."""
    index = case.edge_index
    neighbours = [[] for _ in case.pos]
    for a, b in index.T:
        neighbours[a].append(int(b))
    centre = int(np.square(case.pos - [0.9, 0.65]).sum(axis=1).argmin())
    targets = [centre, *neighbours[centre]]
    # The centre point's inputs in readable units: what "mesh only" gets, and what distance and shape add.
    distance, direction = nearest_on_outline(case.pos[[centre]], case.outline)
    example = {"x": float(case.pos[centre, 0]), "y": float(case.pos[centre, 1]),
               "type": next(k for k, v in NODE_LABELS.items() if v == case.node_type[centre]),
               "distance": float(distance[0]), "direction": direction[0].astype(float).tolist(),
               "shape": shape_descriptors(case.outline).astype(float).tolist()}
    graph = {"case": case.id, "pos": case.pos.astype(np.float32).ravel().tolist(),
             "edges": index[:, index[0] < index[1]].T.ravel().tolist(), "outline": np.round(case.outline, 4).tolist(),
             "centre": centre, "targets": targets, "example_inputs": example, "traces": {}}
    model, x, ei, a, norms = load_checkpoint(folder("M3", shown["M3"]) / "model.pt", case)
    trace = trace_messages(model, x, ei, a, targets)
    saved = np.load(folder("M3", shown["M3"]) / "predictions.npz")[case.id]
    if not np.allclose(norms["y"].undo(trace["predicted"]), saved, atol=2e-5, rtol=2e-5):
        raise ValueError(f"Checkpoint prediction differs from saved output: M3, {case.id}")
    with open(out / "trace_M3.bin", "wb") as f:
        for key in ["before", "incoming", "after", "messages"]:
            f.write(trace[key].astype("<f4").tobytes())
    links = hops_from(case, [centre])
    near = np.nonzero((links >= 0) & (links <= len(model.layers)))[0]
    share = round_influence(model, x, ei, a, centre)
    graph["rounds"] = {"nodes": near.tolist(), "share": [[float(f"{v:.3g}") for v in row[near]] for row in share]}
    graph["obstacle_links"] = int(hops_from(case, np.nonzero(case.node_type == OBSTACLE)[0])[centre])
    graph["traces"]["M3"] = {"file": "trace_M3.bin", "seed": shown["M3"], "layers": len(model.layers),
                             "hidden": model.encode_node[0].out_features,
                             "sources": trace["sources"].tolist(), "destinations": trace["destinations"].tolist()}
    (out / "graph.json").write_text(json.dumps(graph, separators=(",", ":"), allow_nan=False))


def influence(model, x, edge_index, edge_attr, node: int) -> np.ndarray:
    """Influence distribution of the predicted horizontal velocity at `node` (Xu et al. 2018, Definition 3.1):
    each node's influence score is the sum of the absolute Jacobian entries for its inputs, divided by the
    sum over all nodes."""
    score = np.abs(sensitivity(model, x, edge_index, edge_attr, node)).sum(axis=1)
    return score / score.sum()


def export_reach(case: Case, folder, shown: dict, out: Path) -> None:
    """The complete mesh of one test airfoil for the reach and influence slides: each point's link count to
    the airfoil, the wake point the replay starts from, and each network's influence distribution at that
    point with its mean link distance."""
    index = case.edge_index
    start = int(np.square(case.pos - [4.0, 0.0]).sum(axis=1).argmin())  # a point in the wake
    links = hops_from(case, [start])
    reach = {"case": case.id, "pos": case.pos.astype(np.float32).ravel().tolist(),
             "edges": index[:, index[0] < index[1]].T.ravel().tolist(), "outline": case.outline.tolist(),
             "obstacle_hops": hops_from(case, np.nonzero(case.node_type == OBSTACLE)[0]).tolist(), "start": start,
             "influence": {}}
    for s in INPUT_SETS:
        model, x, ei, a, _ = load_checkpoint(folder(s, shown[s]) / "model.pt", case)
        share = influence(model, x, ei, a, start)
        inside = np.nonzero(share)[0]  # exactly zero outside the 12-link receptive field
        reach["influence"][s] = {"nodes": inside.tolist(), "share": [float(f"{v:.3g}") for v in share[inside]],
                                 "mean_links": round(float((share * links).sum()), 2)}
    (out / "reach.json").write_text(json.dumps(reach, separators=(",", ":"), allow_nan=False))


WAKE_POINTS = [(1.5, 0.0), (3.0, 0.3), (5.0, -0.2)]  # behind the obstacle, which is centred at the origin


def influence_distance(cases: list[Case], folder, runs: dict) -> dict:
    """Mean link distance of the influence distribution for each input set, over the given test shapes,
    the wake points above and every seed: [lowest, mean, highest]."""
    spread = {}
    for s, rs in runs.items():
        values = []
        for case in cases:
            for point in WAKE_POINTS:
                node = int(np.square(case.pos - point).sum(axis=1).argmin())
                links = hops_from(case, [node])
                for r in rs:
                    model, x, ei, a, _ = load_checkpoint(folder(s, r["seed"]) / "model.pt", case)
                    values.append(float((influence(model, x, ei, a, node) * links).sum()))
        spread[s] = [round(min(values), 2), round(float(np.mean(values)), 2), round(max(values), 2)]
    return spread


def timing() -> dict | None:
    """Prediction time of one network against SU2's solve time, from lab/runs/timing.json
    (python scripts/timing.py), or None when absent."""
    path = ROOT / "lab" / "runs" / "timing.json"
    return json.loads(path.read_text()) if path.exists() else None


def rounds_experiment() -> dict | None:
    """Velocity error with 6, 12 and 18 message-passing rounds (seed 0, checkpoint chosen on validation).

    Read from lab/runs/rounds_6, rounds_12 and rounds_18 (python flowgnn.py --config examples/rounds_<n>.json),
    which differ only in the number of rounds. Returns None when any of those runs is missing, and the page
    then omits the result.
    """
    names = {k: f"rounds_{k}" for k in (6, 12, 18)}
    paths = {k: ROOT / "lab" / "runs" / v / "summary.json" for k, v in names.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    out = {"seed": 0, "rounds": list(names)}
    for k, path in paths.items():
        study = json.loads(path.read_text())
        if study["settings"]["model"]["layers"] != k:
            raise ValueError(f"{path} does not use {k} rounds")
        for r in study["runs"]:
            group = lambda split: float(np.mean([c["all"]["velocity"] for c in r["cases"].values() if c["split"] == split]))
            entry = out.setdefault(r["input_set"], {"val": [], "test": []})
            entry["val"].append(round(group("val"), 4))
            entry["test"].append(round(group("test"), 4))
    return out


def main() -> None:
    """Generate all page data from one finished study, preserving its reported metrics."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", nargs="?", default="train", help="the run folder in lab/runs/ (default: train)")
    args = parser.parse_args()
    run = ROOT / "lab" / "runs" / args.name
    summary = json.loads((run / "summary.json").read_text())
    settings = summary["settings"]
    out = ROOT / "docs" / "data"
    out.mkdir(parents=True, exist_ok=True)

    runs = {s: [r for r in summary["runs"] if r["input_set"] == s] for s in INPUT_SETS}
    if any(not runs[s] for s in INPUT_SETS):
        raise ValueError("The page requires completed runs for M0, M1, M2 and M3")
    seeds = sorted(r["seed"] for r in runs["M0"])
    if any(sorted(r["seed"] for r in group) != seeds for group in runs.values()):
        raise ValueError("All four models must use the same study seeds")
    for old in [*out.glob("*.bin"), *out.glob("*.json")]:
        old.unlink()
    test = lambda r: [c for c in r["cases"].values() if c["split"] == "test"]  # noqa: E731
    mean_test = lambda r: float(np.mean([c["all"]["velocity"] for c in test(r)]))  # noqa: E731
    # The displayed network per input set is the median seed by validation error, so test data choose nothing.
    mean_val = lambda r: float(np.mean([c["all"]["velocity"] for c in r["cases"].values() if c["split"] == "val"]))  # noqa: E731
    shown = {s: sorted(rs, key=mean_val)[len(rs) // 2]["seed"] for s, rs in runs.items()}
    folder = lambda s, seed: run / f"{s}_seed{seed}"  # noqa: E731
    all_predictions = {(s, r["seed"]): dict(np.load(folder(s, r["seed"]) / "predictions.npz"))
                       for s, rs in runs.items() for r in rs}
    predictions = {s: all_predictions[(s, shown[s])] for s in runs}

    solve_seconds = {r["id"]: r["seconds"] for r in json.loads((ROOT / "data" / "manifest.json").read_text())["cases"]}
    held_out = [c for c in (Case.load(p) for p in sorted((ROOT / "data" / "cases").glob("*.npz"))) if c.split == "test"]
    hop_error = {s: np.zeros((len(HOP_BINS) - 1, 2)) for s in runs}  # sum of |error|, of |disturbance|
    cases, windows = [], {}
    for case in held_out:
        # Error against links from the obstacle, summed over every seed.
        bins = np.digitize(hops_from(case, np.nonzero(case.node_type == OBSTACLE)[0]), HOP_BINS) - 1
        disturbance = np.linalg.norm(case.target[:, :2] - [1.0, 0.0], axis=1)
        for (s, _), pred in all_predictions.items():
            err = np.linalg.norm(pred[case.id][:, :2] - case.target[:, :2], axis=1)
            for b in range(len(HOP_BINS) - 1):
                hop_error[s][b] += err[bins == b].sum(), disturbance[bins == b].sum()

        # Keep the triangles inside the window, renumbering their nodes.
        x0, y0, x1, y1 = WAKE_WINDOW
        centre = case.pos[case.tri].mean(axis=1)
        keep = (centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0) & (centre[:, 1] < y1)
        nodes = np.unique(case.tri[keep])
        renumber = np.full(len(case.pos), -1)
        renumber[nodes] = np.arange(len(nodes))
        reference = case.target[nodes]
        # Colour bounds per field: wide enough for the reference and every shown prediction.
        shown_fields = [reference] + [p[case.id][nodes] for p in predictions.values()]
        ranges = {"reference": [[float(min(f[:, k].min() for f in shown_fields)), float(max(f[:, k].max() for f in shown_fields))]
                                for k in range(3)]}
        errors = {s: np.abs(predictions[s][case.id][nodes] - reference) for s in runs}
        # One error cap per field, shared by every input set, so their colours compare directly.
        ranges["error"] = [float(max(np.percentile(e[:, k], 99) for e in errors.values())) for k in range(3)]
        windows[case.id] = (nodes, ranges)
        with open(out / f"{case.id}.bin", "wb") as f:
            f.write(case.pos[nodes].astype(np.float32).tobytes())
            f.write(renumber[case.tri[keep]].astype("<u4").tobytes())
            for k in range(3):
                f.write(reference[:, k].astype("<f4").tobytes())
            for s in runs:
                for k in range(3):
                    f.write(predictions[s][case.id][nodes, k].astype("<f4").tobytes())
            f.write(nodes.astype("<u4").tobytes())
        cases.append({"id": case.id, "split": case.split, "shape": case.params["shape"], "params": case.params,
                      "su2_seconds": solve_seconds[case.id],
                      "nodes": len(nodes), "triangles": int(keep.sum()), "ranges": ranges,
                      "outline": np.round(case.outline, 4).tolist(),
                      "errors": {s: [{"seed": r["seed"], "all": r["cases"][case.id]["all"]} for r in runs[s]] for s in runs}})

    # The learning replay: the two shown networks retrained with identical settings by examples/watch.json,
    # which records the prediction every 2 epochs (the study runs record every 10).
    watch = ROOT / "lab" / "runs" / "watch"
    watched = {s: watch / f"{s}_seed{shown[s]}" for s in ("M0", "M3")}
    if any(not (f / "snapshots.npz").exists() for f in watched.values()):
        raise FileNotFoundError(f"Train the replay first: python flowgnn.py --config examples/watch.json (seeds {shown})")
    snap_case = json.loads((watch / "summary.json").read_text())["settings"]["evaluation"]["snapshot_case"]
    nodes, _ = windows[snap_case]
    progress = {"case": snap_case, "runs": {}}
    frames = {}
    for s, f in watched.items():
        snap = np.load(f / "snapshots.npz")
        frames[s] = snap["predicted"].astype(np.float32)[:, nodes, 0]  # horizontal velocity, (snapshots, nodes)
        if progress.setdefault("epochs", snap["epochs"].tolist()) != snap["epochs"].tolist():
            raise ValueError("Models must use the same recorded snapshot epochs")
        metrics = json.loads((f / "metrics.json").read_text())
        original = next(r for r in runs[s] if r["seed"] == shown[s])
        progress["runs"][s] = {"seed": shown[s],
                               "velocity": [round(p["cases"][snap_case]["velocity"], 4) for p in metrics["snapshots"]],
                               "test_velocity": round(mean_test(metrics), 4), "study_test_velocity": round(mean_test(original), 4)}
    # 16-bit frames over one shared range: about 5e-5 in velocity per step, far below any colour change.
    low = float(min(v.min() for v in frames.values()))
    high = float(max(v.max() for v in frames.values()))
    progress["range"] = [low, high]
    for s, v in frames.items():
        np.round((v - low) / (high - low) * 65535).astype("<u2").tofile(out / f"progress_{s}.bin")
    graph_case = next(c for c in held_out if c.id == snap_case)
    export_graph(graph_case, folder, shown, out)
    export_reach(next(c for c in held_out if c.id == REACH_CASE), folder, shown, out)
    data = {"version": 3, "input_sets": {s: list(INPUT_SETS[s]) for s in runs}, "cases": cases,
            "hop_bins": HOP_BINS,
            "hop_error": {s: (v[:, 0] / np.maximum(v[:, 1], 1e-12)).round(4).tolist() for s, v in hop_error.items()},
            "epochs": settings["training"]["epochs"], "progress": progress, "rounds_experiment": rounds_experiment(),
            "timing": timing(),
            "influence_distance": influence_distance([c for c in held_out if c.id in (snap_case, REACH_CASE)], folder, runs)}
    (out / "summary.json").write_text(json.dumps(data, separators=(",", ":")))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"Wrote {len(cases)} cases to docs/data/ ({size / 1024:.0f} KB); shown seeds {shown}")


if __name__ == "__main__":
    main()
