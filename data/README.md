# flowgnn data

Steady, incompressible, laminar flow past one obstacle in a channel, solved with SU2 8.5.0.
One `.npz` file per case in `cases/`; `manifest.json` records every setting and how each case
converged; `validation.json` is the check of the solver setup on a circle. Both are made by
[`scripts/data/`](../scripts/data/): `make_data.py` (settings in `dataset.json`, solver settings in
`su2.cfg`) and `validate_circle.py`, with SU2 8.5.0 and `requirements-data.txt`.

## How the data was made

| | |
|---|---|
| Domain | x from -20 to 40, y from -20 to 20; obstacle near the origin |
| Flow | uniform inflow U = 1, density 1, viscosity 0.025, so Re = U L / nu = 40 with L = 1 (the circle's diameter, an airfoil's chord) |
| Steady regime | every obstacle is at most 1 tall across the flow, so Re on frontal height, U h / nu, is at most 40 (from 4.8 to 40.0; the circle is exactly 40) |
| Boundaries | velocity inlet (left), zero gauge pressure outlet (right), slip walls (top and bottom), no-slip obstacle |
| Obstacles | ellipses, rectangles, equilateral triangles and NACA 4-digit airfoils (chord 1) |
| Solve mesh | gmsh triangles, 0.0125 at the obstacle growing linearly to 0.5 at 10 units away, and at most 0.05 in a wake box from 1 unit before the obstacle to 8 units behind it and 1.5 units to either side (blending over 2 units); 34,860 to 40,693 nodes |
| Solver | SU2 8.5.0, incompressible Navier-Stokes (`INC_NAVIER_STOKES`), second-order upwind (FDS + MUSCL, no limiter), Green-Gauss gradients, implicit pseudo-time stepping at CFL 50, FGMRES with ILU to 1e-6. The OpenMP build, one process per case with 6 threads (`SU2_CFD -t 6`), not MPI |
| Converged | log10 of the RMS pressure residual below -9 within at most 20,000 iterations; every case reached it, in 422 to 3,908 iterations (median 515.5) and 78 to 1,203 s |
| Training mesh | gmsh triangles, 0.05 at the obstacle growing linearly to 1.0 at 8 units away; 4,224 to 4,565 nodes. This is the mesh the network reads |
| Transfer | u, v, p interpolated linearly from the solve mesh to the training-mesh nodes; velocity on the obstacle set to exactly 0. Obstacle nodes on the straight walls of rectangles and triangles lie exactly on the solve mesh's boundary edges and can count as outside it; those 510 nodes (of 219,572) take the pressure of the nearest solve node |

All quantities are dimensionless: lengths in units of L, velocity in units of U, pressure in
units of rho U^2 (gauge, zero at the outlet).

Why two meshes: solved directly on the training mesh, the circle's recirculation bubble is 19%
shorter and separation 6.5° further back than on the solve mesh (table below). The training mesh is coarse in the
wake, and refining it would make every graph about nine times larger, so the flow is solved on the
fine mesh and only stored on the coarse one.

## Solver check: the circle at Re 40

`validation.json` holds this check. The circle (`ellipse_01`, diameter 1) is solved with the
settings above on three solve meshes, every cell 2x larger, as used, and 2x smaller, and once
directly on the training mesh. Wake length is measured along the centreline from the back of
the circle to where u changes sign; the separation angle from the front stagnation point to
where the velocity along the wall changes sign (read 0.002 from the wall).

| Mesh or source | Nodes | Cd | Lw/D | separation |
|---|---:|---:|---:|---:|
| training mesh, solved directly | 4,492 | 1.549 | 1.823 | 134.2° |
| solve mesh, cells 2x larger | 10,181 | 1.527 | 2.225 | 127.8° |
| solve mesh, as used | 39,843 | 1.534 | 2.255 | 127.7° |
| solve mesh, cells 2x smaller | 157,273 | 1.538 | 2.265 | 126.9° |
| Dennis & Chang 1970, https://doi.org/10.1017/S0022112070001428 | | 1.522 | 2.345 | 126.2° |
| Gautier, Biau & Lamballais 2013, https://arxiv.org/abs/1310.6641 | | 1.493 | 2.236 | 126.4° |
| range of 17 published results (Gautier et al., Table 1) | | 1.48-1.62 | 2.13-2.35 | 124.4-127.3° |

The published values are for an unbounded domain; this channel's side walls are 20 diameters
away (2.5% blockage), which raises the drag by a few per cent. Drag and wake length on the mesh as
used are inside the published spread. The separation angle, 127.7°, is 0.4° above its highest
published value (127.3°); the finer mesh gives 126.9° and the extrapolated value is 126.7°, both
inside the spread, so most of that difference is discretisation error. The grid convergence index
(Celik et al. 2008, https://doi.org/10.1115/1.2960953) from the three solve meshes:

| Quantity | Observed order | Extrapolated | GCI, mesh as used | GCI, finest mesh |
|---|---:|---:|---:|---:|
| Cd | 0.72 | 1.5449 | 0.89% | 0.54% |
| Lw/D | 1.72 | 2.2689 | 0.75% | 0.23% |
| separation | 2.38 | 126.69° | 0.98% | 0.19% |

Before running, I set these bands to pass: drag 1.49-1.57, wake length 2.15-2.35, separation
angle 125°-128°, and a grid convergence index below 2% on the mesh used for the data.
All pass. `tests/test_data.py` checks them, plus that the circle stored on the training mesh
ends its bubble within half an edge of the solve mesh's value: read by straight lines between
training-mesh nodes it is 2.09 diameters long, because the edges there are 0.3 long.

## Splits

50 cases: 14 ellipses, 14 rectangles, 16 airfoils and 6 triangles.

- **train** (34): 12 ellipses (one is the circle), 12 rectangles and 10 airfoils, at
  random sizes, angles and positions near the origin.
- **val** (6): 2 ellipses, 2 rectangles, 2 airfoils.
- **test** (10): the 6 triangles, a shape type that is not in the training data, and 4
  airfoils with thickness, camber and angle of attack combinations that are not in training
  (NACA 3415 at 8°, 0009 at 14°, 5410 at 3°, 2421 at -6°).

Airfoils use the standard NACA 4-digit formulas with the trailing edge closed to a point; a
positive angle of attack lifts the nose. Each case's full definition is in its `params`.

## File contents (`cases/<id>.npz`)

| Array | Shape | Meaning |
|---|---|---|
| `pos` | (N, 2) float64 | training-mesh node coordinates |
| `tri` | (T, 3) int32 | triangles (node indices) |
| `node_type` | (N,) int8 | 0 fluid, 1 inlet, 2 outlet, 3 side wall, 4 obstacle (`src/graph.py`, `NODE_LABELS`) |
| `outline` | (M, 2) float64 | obstacle boundary polygon, counter-clockwise |
| `U` | (N, 2) float32 | velocity |
| `p` | (N,) float32 | pressure |
| `params` | string | JSON of the case definition |

`manifest.json` stores the full generation settings (`config`), the SU2 and gmsh versions, and
per case its Reynolds number on frontal height, both mesh sizes, iterations, final residuals,
solver time, and drag and lift coefficients (from the solve mesh).

```python
import numpy as np
case = np.load("data/cases/triangle_05.npz")
print(case["pos"].shape, case["U"].shape)
```
