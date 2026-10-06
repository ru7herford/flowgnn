"""Would more training shapes help? Train M0 and M3 (seed 0) on half the training shapes.

Same settings as examples/rounds_12.json, but only every other training shape of each family (sorted by id)
is kept; validation and test shapes are unchanged. Compare its validation errors with lab/runs/rounds_12:
if halving the data barely changed them, more data would be unlikely to help.

    python scripts/half_data.py      # writes lab/runs/half_data/
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowgnn import train_one  # noqa: E402
from src.graph import Case  # noqa: E402

cfg = json.loads((ROOT / "examples" / "rounds_12.json").read_text())
cases = [Case.load(p) for p in sorted((ROOT / "data" / "cases").glob("*.npz"))]
by_family = defaultdict(list)
for c in cases:
    if c.split == "train":
        by_family[c.params["shape"]].append(c)
kept = {c.id for family in by_family.values() for c in sorted(family, key=lambda c: c.id)[::2]}
cases = [c for c in cases if c.split != "train" or c.id in kept]
print(f"training on {len(kept)} of {sum(len(f) for f in by_family.values())} training shapes", flush=True)
cases += [c.mirrored() for c in cases if c.split == "train"]  # as in flowgnn.py
device = "cuda" if torch.cuda.is_available() else "cpu"
run_dir = ROOT / "lab" / "runs" / "half_data"
results = [train_one(cases, s, 0, cfg, run_dir / f"{s}_seed0", device) for s in ("M0", "M3")]
(run_dir / "summary.json").write_text(json.dumps({"settings": cfg, "kept": sorted(kept), "runs": results}, indent=2) + "\n")
