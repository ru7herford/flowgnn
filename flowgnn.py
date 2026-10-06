"""FlowGNN: which inputs does a graph network need to predict flow around unseen shapes?

Trains the same network once per input set (M0-M3, see src/graph.py) and seed listed in
the settings file, evaluates every model on the validation and test cases, and writes
everything to lab/runs/<settings name>/:

    <input set>_seed<k>/history.csv      loss per epoch (train, and validation every few epochs)
    <input set>_seed<k>/metrics.json     errors per validation/test case, and on the test cases at every snapshot
    <input set>_seed<k>/predictions.npz  predicted u, v, p per validation/test case
    <input set>_seed<k>/snapshots.npz    predicted u, v, p on one test case at every snapshot epoch
    <input set>_seed<k>/model.pt         weights and normalisation
    summary.json                         all metrics, plus the settings used

Every training shape is used twice: as solved and as its top-to-bottom mirror image. The saved weights are
those of the epoch with the lowest validation loss, so test shapes never choose a checkpoint. A settings
file may list "runs" (input set and seed pairs) instead of every set with every seed.

    python flowgnn.py                          # every input set and seed
    python flowgnn.py --sets M3 --seeds 0      # one model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.graph import INPUT_SETS, Case, build_graphs
from src.model import FlowGNN

ROOT = Path(__file__).resolve().parent


def batches(graphs, size, generator):
    """Several graphs joined into one disconnected graph, in a new random order every epoch."""
    order = torch.randperm(len(graphs), generator=generator).tolist()
    for start in range(0, len(order), size):
        chosen = [graphs[i] for i in order[start:start + size]]
        offsets = np.cumsum([0] + [len(g.x) for g in chosen[:-1]])
        yield (torch.cat([g.x for g in chosen]),
               torch.cat([g.edge_index + int(o) for g, o in zip(chosen, offsets)], dim=1),
               torch.cat([g.edge_attr for g in chosen]),
               torch.cat([g.y for g in chosen]))


def errors(case: Case, predicted: np.ndarray) -> dict:
    """Relative L2 errors of velocity and pressure over all mesh nodes (stored under "all").

    Velocity errors are relative to the disturbance the obstacle causes, u - (1, 0):
    far from the obstacle u is close to 1, so an error relative to u itself would
    look small even for a network that had learned nothing. Pressure errors are relative to p itself.
    """
    true = case.target
    disturbance = true[:, :2] - [1.0, 0.0]
    velocity = np.linalg.norm(predicted[:, :2] - true[:, :2]) / np.linalg.norm(disturbance)
    pressure = np.linalg.norm(predicted[:, 2] - true[:, 2]) / np.linalg.norm(true[:, 2])
    return {"all": {"velocity": float(velocity), "pressure": float(pressure)}}


def snapshot_epochs(every: int, epochs: int) -> list[int]:
    """Untrained, after epochs 1 and 2, then every `every` epochs: fast early learning and later plateaus both show."""
    return sorted({0, 1, 2, *range(every, epochs + 1, every), epochs})


def train_one(cases, input_set, seed, cfg, out: Path, device: str) -> dict:
    """Train one input set and seed, saving curves, checkpoints and held-out predictions."""
    torch.manual_seed(seed)
    graphs, norms = build_graphs(cases, input_set, device)
    train = [g for g in graphs if g.case.split == "train"]
    held_out = [g for g in graphs if g.case.split != "train"]
    val = [g for g in held_out if g.case.split == "val"]
    test = [g for g in held_out if g.case.split == "test"]

    m, t, e = cfg["model"], cfg["training"], cfg["evaluation"]
    model = FlowGNN(graphs[0].x.shape[1], graphs[0].edge_attr.shape[1], graphs[0].y.shape[1],
                    m["hidden"], m["layers"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], fused=device == "cuda")
    # One-cycle learning rate: it rises for the first 30% of the steps, then decays (PyTorch's defaults).
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=t["learning_rate"],
        total_steps=t["epochs"] * math.ceil(len(train) / t["batch_cases"]))
    generator = torch.Generator().manual_seed(seed)

    # On a GPU the training step runs in bfloat16 with the loss in float32, compiled once per run
    # (the node count changes from batch to batch, hence dynamic shapes). Validation, snapshots and
    # the final predictions run in float32 without compiling.
    fast = device == "cuda"

    def step_loss(x, edge_index, edge_attr, y):
        """Evaluate mean squared error, keeping the loss in float32 on the GPU."""
        with torch.autocast("cuda", torch.bfloat16, enabled=fast):
            predicted = model(x, edge_index, edge_attr)
        return torch.mean((predicted.float() - y) ** 2)

    if fast:
        torch._dynamo.reset()  # a fresh compile per run, so no run inherits another's cached graphs
        step_loss = torch.compile(step_loss, dynamic=True)

    def predict(g) -> np.ndarray:
        """Decode a complete graph and restore the original flow units."""
        return norms["y"].undo(model(g.x, g.edge_index, g.edge_attr).cpu().numpy())

    shown = next((g for g in test if g.case.id == e["snapshot_case"]), None)
    snap_at = set(snapshot_epochs(e["snapshot_every"], t["epochs"]))
    snapshots, snapshot_errors = [], []

    def take_snapshot(epoch: int) -> None:
        """Predict every test case once, without updating weights: keep the shown case's field and
        each case's errors."""
        model.eval()
        with torch.no_grad():
            predicted = {g.case.id: predict(g) for g in test}
        model.train()
        if shown is not None:
            snapshots.append(predicted[shown.case.id].astype(np.float16))
        scores = {g.case.id: errors(g.case, predicted[g.case.id])["all"] for g in test}
        snapshot_errors.append({"epoch": epoch, "cases": scores})

    out.mkdir(parents=True, exist_ok=True)
    history = []
    # The saved weights are those of the epoch with the lowest validation loss, so test cases never choose a
    # checkpoint.
    if not val:
        raise ValueError("Choosing the saved weights requires validation cases")
    best_validation, selected_epoch, selected_weights = float("inf"), t["epochs"], None
    n_train = sum(len(g.y) for g in train)
    take_snapshot(0)
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = torch.zeros((), device=device)
        for x, edge_index, edge_attr, y in batches(train, t["batch_cases"], generator):
            loss = step_loss(x, edge_index, edge_attr, y)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimiser.step()
            schedule.step()
            total += loss.detach() * len(y)
        row = {"epoch": epoch, "train_mse": total.item() / n_train, "val_mse": ""}  # the one host sync per epoch
        if epoch == 1:
            first_epoch = time.perf_counter() - start
        if epoch % t["validate_every"] == 0 or epoch == t["epochs"]:
            # Compare the same normalised data error; test cases never choose a checkpoint.
            model.eval()
            with torch.no_grad():
                losses = torch.stack([torch.mean((model(g.x, g.edge_index, g.edge_attr) - g.y) ** 2) for g in val])
            row["val_mse"] = float(np.mean(losses.cpu().numpy().astype(np.float64)))  # one transfer for all cases
            if row["val_mse"] < best_validation:
                best_validation, selected_epoch = row["val_mse"], epoch
                selected_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            print(f"    epoch {epoch:4d}  train {row['train_mse']:.2e}  val {row['val_mse']:.2e}", flush=True)
        if epoch in snap_at:
            take_snapshot(epoch)
        history.append(row)
    seconds = time.perf_counter() - start

    model.load_state_dict(selected_weights)
    model.eval()
    metrics, predictions = {}, {}
    with torch.no_grad():
        for g in held_out:
            predicted = predict(g)
            predictions[g.case.id] = predicted.astype(np.float32)
            metrics[g.case.id] = {"split": g.case.split, "shape": g.case.params["shape"],
                                  **errors(g.case, predicted)}

    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(out / "predictions.npz", **predictions)
    if snapshots:
        np.savez_compressed(out / "snapshots.npz", case=shown.case.id, epochs=np.array(sorted(snap_at), dtype=np.int16),
                            predicted=np.stack(snapshots))
    torch.save({"weights": model.state_dict(), "normalisation": {k: n.state() for k, n in norms.items()},
                "input_set": input_set, "model": m}, out / "model.pt")
    result = {"input_set": input_set, "seed": seed, "train_seconds": round(seconds, 1),
              "first_epoch_seconds": round(first_epoch, 1), "final_val_mse": history[-1]["val_mse"],
              "selected_epoch": selected_epoch,
              "selected_val_mse": best_validation,
              "cases": metrics, "snapshots": snapshot_errors}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"    {seconds:.0f} s (first epoch, with compiling, {first_epoch:.0f} s)", flush=True)
    return result


def report(results: list[dict]) -> None:
    """Mean relative error per input set on each group of test shapes, with the spread over seeds."""
    groups = sorted({c["shape"] for r in results for c in r["cases"].values() if c["split"] == "test"})
    print("\nVelocity error on test cases (relative to the disturbance), mean over cases (min-max over seeds):")
    print(f"{'inputs':8s}" + "".join(f"{g:>22s}" for g in groups))
    for input_set in INPUT_SETS:
        runs = [r for r in results if r["input_set"] == input_set]
        if not runs:
            continue
        cells = []
        for group in groups:
            per_seed = [np.mean([c["all"]["velocity"] for c in r["cases"].values()
                                 if c["split"] == "test" and c["shape"] == group]) for r in runs]
            cells.append(f"{np.mean(per_seed):8.2%} ({min(per_seed):.1%}-{max(per_seed):.1%})")
        print(f"{input_set:8s}" + "".join(f"{c:>22s}" for c in cells))


def main() -> None:
    """Run the configured input sets and seeds, then collect their measured results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "train.json")
    parser.add_argument("--sets", nargs="+", choices=list(INPUT_SETS), help="input sets to train (default: all in the settings)")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds to train (default: all in the settings)")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = [Case.load(p) for p in sorted((ROOT / "data" / "cases").glob("*.npz"))]
    counts = {s: sum(c.split == s for c in cases) for s in ("train", "val", "test")}
    print(f"{len(cases)} cases ({counts['train']} train, {counts['val']} val, {counts['test']} test) on {device}")
    cases += [c.mirrored() for c in cases if c.split == "train"]
    print(f"Training on {2 * counts['train']} graphs: each training case and its top-to-bottom mirror image")

    run_dir = ROOT / "lab" / "runs" / args.config.stem
    results = []
    pairs = ([(r["input_set"], r["seed"]) for r in cfg["runs"]] if "runs" in cfg and not (args.sets or args.seeds)
             else [(s, k) for s in args.sets or cfg["input_sets"] for k in args.seeds or cfg["seeds"]])
    for input_set, seed in pairs:
        print(f"  {input_set}, seed {seed}", flush=True)
        results.append(train_one(cases, input_set, seed, cfg, run_dir / f"{input_set}_seed{seed}", device))
    (run_dir / "summary.json").write_text(json.dumps({"settings": cfg, "device": device, "runs": results}, indent=2) + "\n")
    report(results)
    print(f"\nResults in {run_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
