"""Check the page data against the dataset and graph arithmetic.

These checks need no trained weights. Model replay is checked separately in
test_inspect.py; this file protects the exported binary layout and the shown inputs.
"""

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import NODE_LABELS, Case, nearest_on_outline, shape_descriptors  # noqa: E402
DATA = ROOT / "docs" / "data"


def graph_distances(adjacency, starts):
    """Measure shortest connection counts independently of the browser implementation."""
    distance = np.full(len(adjacency), -1)
    pending = deque(starts)
    distance[list(pending)] = 0
    while pending:
        source = pending.popleft()
        for target in adjacency[source]:
            if distance[target] < 0:
                distance[target] = distance[source] + 1
                pending.append(target)
    return distance


def test_fields_match_dataset_and_have_complete_layout():
    """Stable IDs must recover the original positions, reference fields and triangles."""
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["version"] == 3
    assert list(summary["input_sets"]) == ["M0", "M1", "M2", "M3"]
    for record in summary["cases"]:
        count, triangles = record["nodes"], record["triangles"]
        content = (DATA / f"{record['id']}.bin").read_bytes()
        positions = np.frombuffer(content, "<f4", 2 * count).reshape(-1, 2)
        indices = np.frombuffer(content, "<u4", 3 * triangles, 8 * count)
        offset = 8 * count + 12 * triangles
        fields = np.frombuffer(content, "<f4", 15 * count, offset).reshape(5, 3, count)
        offset += 60 * count
        node_ids = np.frombuffer(content, "<u4", count, offset)
        assert len(content) == offset + 4 * count
        original = np.load(ROOT / "data" / "cases" / f"{record['id']}.npz")
        np.testing.assert_allclose(positions, original["pos"][node_ids])
        np.testing.assert_allclose(fields[0].T, np.column_stack([original["U"], original["p"]])[node_ids])
        assert indices.max() < count
        assert np.isfinite(fields).all()
        for component, (low, high) in enumerate(record["ranges"]["reference"]):
            assert fields[:, component].min() >= low - 1e-6
            assert fields[:, component].max() <= high + 1e-6
    progress = summary["progress"]
    nodes = next(c["nodes"] for c in summary["cases"] if c["id"] == progress["case"])
    epochs = progress["epochs"]
    assert epochs == sorted(set(epochs))
    low, high = progress["range"]
    assert low < high
    for model in progress["runs"]:
        frames = np.fromfile(DATA / f"progress_{model}.bin", dtype="<u2")
        assert frames.size == len(epochs) * nodes
        assert frames.min() == 0 or frames.max() == 65535  # the range is the data's own extent
        assert len(progress["runs"][model]["velocity"]) == len(epochs)


def adjacency_of(data):
    """Neighbour lists from an exported edge list (each link stored once)."""
    adjacency = [[] for _ in range(len(data["pos"]) // 2)]
    for source, target in np.array(data["edges"]).reshape(-1, 2):
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def test_full_graph_traces_and_inputs():
    """Check every recorded sum and the shown example inputs."""
    graph = json.loads((DATA / "graph.json").read_text())
    adjacency = adjacency_of(graph)
    # The stored graph must cover the complete dataset, including outside the drawing window.
    original = np.load(ROOT / "data" / "cases" / f"{graph['case']}.npz")
    assert len(adjacency) == len(original["pos"])
    np.testing.assert_allclose(np.array(graph["pos"]).reshape(-1, 2), original["pos"])
    for model, info in graph["traces"].items():
        rounds, width = info["layers"], info["hidden"]
        slots = len(graph["targets"])
        states = rounds * slots * width
        flat = np.fromfile(DATA / info["file"], dtype="<f4")
        assert flat.size == 3 * states + rounds * len(info["sources"]) * width
        before, incoming, after = flat[:3 * states].reshape(3, rounds, slots, width)
        messages = flat[3 * states:].reshape(rounds, -1, width)
        destinations = np.array(info["destinations"])
        assert np.isfinite(flat).all()
        np.testing.assert_array_equal(before[1:], after[:-1])
        for slot, target in enumerate(graph["targets"]):
            sources = np.array(info["sources"])[destinations == target]
            assert set(sources) == set(adjacency[target])
            total = messages[:, destinations == target].sum(axis=1)
            np.testing.assert_allclose(total, incoming[:, slot], atol=3e-5, rtol=2e-5)
    # The example row is recomputed here from the dataset with the training code's own functions.
    case = Case.load(ROOT / "data" / "cases" / f"{graph['case']}.npz")
    centre, example = graph["centre"], graph["example_inputs"]
    distance, direction = nearest_on_outline(case.pos[[centre]], case.outline)
    np.testing.assert_allclose([example["x"], example["y"]], case.pos[centre], rtol=1e-6)
    assert NODE_LABELS[example["type"]] == case.node_type[centre]
    np.testing.assert_allclose(example["distance"], distance[0], rtol=1e-6)
    np.testing.assert_allclose(example["direction"], direction[0], rtol=1e-6)
    np.testing.assert_allclose(example["shape"], shape_descriptors(case.outline), rtol=1e-6)


def test_reach_mesh():
    """The reach page's airfoil: complete mesh, link counts from the airfoil, and a start beyond 12 links."""
    reach = json.loads((DATA / "reach.json").read_text())
    adjacency = adjacency_of(reach)
    original = np.load(ROOT / "data" / "cases" / f"{reach['case']}.npz")
    np.testing.assert_allclose(np.array(reach["pos"]).reshape(-1, 2), original["pos"])
    obstacle = np.flatnonzero(original["node_type"] == NODE_LABELS["obstacle"])
    np.testing.assert_array_equal(graph_distances(adjacency, obstacle), reach["obstacle_hops"])
    graph = json.loads((DATA / "graph.json").read_text())
    assert reach["obstacle_hops"][reach["start"]] > graph["traces"]["M3"]["layers"]


if __name__ == "__main__":
    test_fields_match_dataset_and_have_complete_layout()
    print("ok  page fields and saved snapshots")
    test_full_graph_traces_and_inputs()
    print("ok  complete graph, message sums and shown inputs")
    test_reach_mesh()
    print("ok  reach mesh and start point")
