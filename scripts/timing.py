"""How long one trained network takes to predict a test shape, against the time SU2 took to solve it.

Warm: inputs already on the device, after warm-up runs. End to end: reading the case file, building the
inputs, predicting and copying the result back. SU2's times come from data/manifest.json. The network
predicts on the training mesh (about 4,400 nodes); SU2 solved on the finer solve mesh (35-41k nodes).
Writes lab/runs/timing.json for scripts/export_page.py.

    python scripts/timing.py        # after python flowgnn.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import Case, edge_inputs, node_inputs  # noqa: E402
from src.inspect import load_checkpoint  # noqa: E402


def timed(run, repeats: int, device: str) -> float:
    """Median wall time of `run` in seconds, waiting for the GPU to finish each time."""
    for _ in range(5):
        run()
    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def prediction_times(checkpoint: Path, paths: list[Path], device: str) -> dict:
    """Median warm and end-to-end prediction times of one network over the test shapes."""
    warm, end_to_end, nodes = [], [], []
    for path in paths:
        case = Case.load(path)
        model, x, edge_index, edge_attr, norms = load_checkpoint(checkpoint, case, device)
        input_set = torch.load(checkpoint, map_location="cpu", weights_only=True)["input_set"]

        @torch.no_grad()
        def predict_warm():
            model(x, edge_index, edge_attr)

        @torch.no_grad()
        def predict_from_file():
            fresh = Case.load(path)  # a new case object, so the distance inputs are computed again
            xs = torch.as_tensor(norms["x"](node_inputs(fresh, input_set)), device=device)
            es = torch.as_tensor(norms["edge"](edge_inputs(fresh)), device=device)
            index = torch.as_tensor(fresh.edge_index, device=device).long()
            norms["y"].undo(model(xs, index, es).cpu().numpy())

        warm.append(timed(predict_warm, 50, device))
        end_to_end.append(timed(predict_from_file, 10, device))
        nodes.append(len(case.pos))
    return {"warm_seconds": statistics.median(warm), "end_to_end_seconds": statistics.median(end_to_end),
            "nodes": [min(nodes), max(nodes)]}


def main() -> None:
    """Time M3 on every test shape (on the GPU when there is one, and on the CPU) and write the result."""
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    tests = [r["id"] for r in manifest["cases"] if r["split"] == "test"]
    paths = [ROOT / "data" / "cases" / f"{case_id}.npz" for case_id in tests]
    su2 = sorted(r["seconds"] for r in manifest["cases"] if r["split"] == "test")
    checkpoint = ROOT / "lab" / "runs" / "train" / "M3_seed0" / "model.pt"  # every input set costs the same
    timing = {"su2_seconds": [su2[0], statistics.median(su2), su2[-1]], "su2": manifest["solver"],
              "gpu": prediction_times(checkpoint, paths, "cuda") if torch.cuda.is_available() else None,
              "cpu": prediction_times(checkpoint, paths, "cpu"), "cpu_threads": torch.get_num_threads(),
              "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    out = ROOT / "lab" / "runs" / "timing.json"
    out.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
