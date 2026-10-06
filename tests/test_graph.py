"""Checks of the network's inputs. Run with `python tests/test_graph.py` (or pytest)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph import NODE_TYPES, Case, edges, nearest_on_outline, node_inputs, shape_descriptors  # noqa: E402

SQUARE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def test_distance_and_direction_to_a_square():
    """Known square distances check both nearest-edge projection and direction."""
    points = np.array([[2.0, 0.0], [0.0, -1.5], [1.5, 1.5]])
    d, direction = nearest_on_outline(points, SQUARE)
    assert np.allclose(d, [1.5, 1.0, np.sqrt(2)])
    assert np.allclose(direction, [[-1, 0], [0, 1], [-np.sqrt(0.5), -np.sqrt(0.5)]])


def test_shape_descriptors_see_orientation():
    """Rotation changes extent, while uneven sampling must not move the area centroid."""
    flat = SQUARE * [2.0, 0.5]                 # 2 long, 0.5 high
    upright = flat[:, ::-1]                    # the same rectangle turned 90 degrees
    assert np.allclose(shape_descriptors(flat), [2.0, 0.5, 1.0, 5.0, 0.0, 0.0])
    assert np.allclose(shape_descriptors(upright)[:2], [0.5, 2.0])
    # The centre is the area centroid: unevenly spaced outline points do not move it.
    crowded = np.array([[-1, -1], [0.8, -1], [0.9, -1], [1, -1], [1, 1], [-1, 1]])
    assert np.allclose(shape_descriptors(crowded)[4:], [0.0, 0.0])


def test_edges_are_unique_and_in_both_directions():
    """Two adjacent triangles must produce ten distinct directed connections."""
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    e = edges(tri)
    pairs = set(map(tuple, e.T))
    assert len(pairs) == e.shape[1] == 10 and all((b, a) in pairs for a, b in pairs)


def test_mirrored_case_is_the_reflected_flow():
    """Reflection reverses vertical velocity while preserving scalar geometry inputs."""
    case = Case.load(ROOT / "data" / "cases" / "naca4412_aoa8.npz")
    mirror = case.mirrored()
    assert np.allclose(mirror.pos[:, 1], -case.pos[:, 1]) and np.allclose(mirror.target[:, 1], -case.target[:, 1])
    assert np.allclose(mirror.target[:, [0, 2]], case.target[:, [0, 2]])
    # The outline stays counter-clockwise, and the distance input is the same at mirrored nodes.
    x, y = mirror.outline.T
    assert 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0
    distance = 2 + NODE_TYPES  # the column after position and boundary type
    assert np.allclose(node_inputs(mirror, "M1")[:, distance], node_inputs(case, "M1")[:, distance], atol=1e-6)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
