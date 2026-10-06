"""Turn a CFD case (data/cases/<id>.npz) into a graph the network can read.

Nodes are mesh nodes and edges are mesh edges, in both directions. What each
node is allowed to know depends on the input set being tested:

    M0   position and boundary type (inlet, outlet, side wall, obstacle)
    M1   M0 + distance and direction to the nearest point of the obstacle
    M2   M0 + six numbers describing the obstacle (length, height, area, perimeter, centre)
    M3   M0 + both

Obstacle boundary labels travel along the mesh, one edge per layer. Position,
mesh structure and any supplied shape features can also provide geometry clues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

INPUT_SETS = {
    "M0": (),                        # mesh only
    "M1": ("distance",),             # + distance
    "M2": ("shape",),                # + shape numbers
    "M3": ("distance", "shape"),     # + both
}
# Boundary type of every node, stored in data/cases/<id>.npz as `node_type`.
NODE_LABELS = {"fluid": 0, "inlet": 1, "outlet": 2, "side wall": 3, "obstacle": 4}
NODE_TYPES = len(NODE_LABELS)
OBSTACLE = NODE_LABELS["obstacle"]
# The near wake around the obstacle (x0, y0, x1, y1), drawn on the page and in the data plots.
WAKE_WINDOW = (-3.0, -3.0, 10.0, 3.0)
POSITION_SCALE = 5.0  # keep positions of order one near the obstacle


@dataclass
class Case:
    """One solved case as plain arrays, before any normalisation."""

    id: str
    split: str
    params: dict
    pos: np.ndarray        # (N, 2)
    tri: np.ndarray        # (T, 3)
    node_type: np.ndarray  # (N,)
    outline: np.ndarray    # (M, 2) obstacle polygon, counter-clockwise
    target: np.ndarray     # (N, 3) u, v, p

    @classmethod
    def load(cls, path: Path) -> "Case":
        """Read one solver case and combine velocity and pressure into prediction targets."""
        data = np.load(path)
        params = json.loads(str(data["params"]))
        target = np.column_stack([data["U"], data["p"]]).astype(np.float32)
        return cls(params["id"], params["split"], params, data["pos"], data["tri"], data["node_type"],
                   data["outline"], target)

    def mirrored(self) -> "Case":
        """The same flow reflected top to bottom: y -> -y and v -> -v.

        The channel, its slip walls and the uniform inflow are symmetric about y = 0,
        so the reflection is an exact solution for the reflected obstacle.
        """
        flip = np.array([1.0, -1.0])
        return Case(self.id + "_mirrored", self.split, self.params, self.pos * flip, self.tri, self.node_type,
                    (self.outline * flip)[::-1], self.target * np.array([1.0, -1.0, 1.0], dtype=np.float32))

    # The mesh never changes, so its geometry is worked out once per case, on first use, and shared
    # by every input set, seed and evaluation.
    @cached_property
    def edge_index(self) -> np.ndarray:
        """Mesh edges as a (2, E) array, each listed in both directions."""
        return edges(self.tri)

    @cached_property
    def to_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
        """Every node's distance to the obstacle surface, and the unit direction towards it."""
        return nearest_on_outline(self.pos, self.outline)


def edges(tri: np.ndarray) -> np.ndarray:
    """Unique mesh edges as a (2, E) array, each listed in both directions."""
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return np.concatenate([pairs, pairs[:, ::-1]]).T


def nearest_on_outline(points: np.ndarray, outline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the obstacle surface, and the unit vector pointing to it."""
    a, b = outline, np.roll(outline, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", points[:, None] - a, ab) / np.einsum("mk,mk->m", ab, ab), 0.0, 1.0)
    closest = a + t[..., None] * ab                       # (N, M, 2) closest point on every side
    gap = closest - points[:, None]
    dist = np.linalg.norm(gap, axis=-1)
    nearest = dist.argmin(axis=1)
    rows = np.arange(len(points))
    d = dist[rows, nearest]
    direction = gap[rows, nearest] / np.maximum(d, 1e-12)[:, None]
    return d, direction


def shape_descriptors(outline: np.ndarray) -> np.ndarray:
    """Streamwise length, frontal height, area, perimeter and centre of the obstacle.

    The flow runs along +x, so length and height depend on how the shape is turned,
    which a rotation-invariant description (such as aspect ratio alone) would miss.
    The centre says where the wake starts.
    """
    x, y = outline[:, 0], outline[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y                      # shoelace terms, one per side
    signed_area = 0.5 * cross.sum()
    perimeter = np.hypot(xn - x, yn - y).sum()
    # Area centroid, not the mean of the outline points: airfoil outlines crowd their points
    # near both edges, which would pull a point average towards them.
    cx = ((x + xn) * cross).sum() / (6 * signed_area)
    cy = ((y + yn) * cross).sum() / (6 * signed_area)
    return np.array([np.ptp(x), np.ptp(y), abs(signed_area), perimeter, cx, cy])


def node_inputs(case: Case, input_set: str) -> np.ndarray:
    """Raw (unnormalised) node inputs for one of the INPUT_SETS."""
    columns = [case.pos / POSITION_SCALE, np.eye(NODE_TYPES)[case.node_type]]
    if "distance" in INPUT_SETS[input_set]:
        d, direction = case.to_obstacle
        columns += [np.log1p(d)[:, None], direction]  # log: resolves the thin layer at the wall and the far field
    if "shape" in INPUT_SETS[input_set]:
        columns.append(np.broadcast_to(shape_descriptors(case.outline), (len(case.pos), 6)))
    return np.column_stack(columns).astype(np.float32)


def edge_inputs(case: Case) -> np.ndarray:
    """Relative position of the neighbour and the edge length, for every edge of case.edge_index."""
    offset = case.pos[case.edge_index[1]] - case.pos[case.edge_index[0]]
    return np.column_stack([offset, np.linalg.norm(offset, axis=1)]).astype(np.float32)


@dataclass
class Graph:
    """Tensors for one case, on one device, ready for the model."""

    case: Case
    x: torch.Tensor           # (N, F) node inputs, normalised
    edge_index: torch.Tensor  # (2, E)
    edge_attr: torch.Tensor   # (E, 3) normalised
    y: torch.Tensor           # (N, 3) targets, normalised


class Normaliser:
    """Mean and standard deviation per column, measured on the training cases only."""

    def __init__(self, arrays: list[np.ndarray]):
        """Fit column scales to the supplied training arrays."""
        stacked = np.concatenate(arrays)
        self.mean = stacked.mean(axis=0)
        self.std = np.maximum(stacked.std(axis=0), 1e-6)

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Convert columns to their training scale as float32."""
        return ((a - self.mean) / self.std).astype(np.float32)

    def undo(self, a: np.ndarray) -> np.ndarray:
        """Restore the physical scale of decoded predictions."""
        return a * self.std + self.mean

    def state(self) -> dict:
        """Save the fitted scales alongside model weights."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "Normaliser":
        """Restore training statistics without fitting them again on held-out data."""
        obj = cls.__new__(cls)
        obj.mean = np.asarray(state["mean"], dtype=np.float32)
        obj.std = np.asarray(state["std"], dtype=np.float32)
        if obj.mean.shape != obj.std.shape or np.any(obj.std <= 0):
            raise ValueError("Invalid saved normalisation statistics")
        return obj


def build_graphs(cases: list[Case], input_set: str, device: str) -> tuple[list[Graph], dict[str, Normaliser]]:
    """Graphs for every case, normalised with statistics from the training cases."""
    raw = [(case, node_inputs(case, input_set), case.edge_index, edge_inputs(case)) for case in cases]
    train = [r for r in raw if r[0].split == "train"]
    norms = {
        "x": Normaliser([r[1] for r in train]),
        "edge": Normaliser([r[3] for r in train]),
        "y": Normaliser([r[0].target for r in train]),
    }
    # Positions and one-hot boundary types are already of order one; leave them as they are
    # so a node's position keeps its meaning across input sets.
    norms["x"].mean[: 2 + NODE_TYPES] = 0.0
    norms["x"].std[: 2 + NODE_TYPES] = 1.0

    tensor = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    graphs = [Graph(case, tensor(norms["x"](x)), tensor(edge_index).long(), tensor(norms["edge"](e)),
                    tensor(norms["y"](case.target)))
              for case, x, edge_index, e in raw]
    return graphs, norms
