"""The shared rugged sea floor (terrain.py): deterministic, bilinear, uneven, calm round the dock, with objects that stand on the bottom."""
import numpy as np
import pytest

from terrain import Terrain, get_terrain

SMALL = {"size_m": 120.0, "dx_m": 0.5, "boulders": 40, "trenches": 2}


def test_same_seed_same_floor_other_seed_other_floor():
    a, b = Terrain(dict(SMALL, seed=5)), Terrain(dict(SMALL, seed=5))
    c = Terrain(dict(SMALL, seed=6))
    assert np.array_equal(a.z, b.z) and np.array_equal(a.refl, b.refl)
    assert not np.allclose(a.z, c.z)                                   # negative control: a different seed is a different sea floor


def test_the_floor_is_very_uneven_but_calm_round_the_dock():
    t = get_terrain({"seed": 5})
    assert t.n == 1200 and abs(t.extent[0] + 150.0) < 1e-6
    z = t.z
    assert z.max() - z.min() > 12.0                                    # more than 12 m of relief over the area
    gx, gy = np.gradient(z, t.dx, t.dx)
    assert 0.25 < float(np.median(np.hypot(gx, gy))) < 0.7             # a rugged floor, not a cliff
    assert abs(t.height(10.0, 0.0) - 11.0) < 0.5                        # the dock keeps its floor ~11 m (dock at 3 m depth)
    near = t.height(*np.meshgrid(np.linspace(0, 20, 20), np.linspace(-10, 10, 20), indexing="ij"))
    assert near.max() - near.min() < 1.5                                # calm patch
    far = t.height(*np.meshgrid(np.linspace(-120, -80, 20), np.linspace(-60, -20, 20), indexing="ij"))
    assert far.max() - far.min() > 3.0                                  # uneven away from it (negative control for the calm patch)


def test_bilinear_height_normal_and_reflectivity():
    t = Terrain(dict(SMALL, relief_m=0.0, ripples_m=0.0, boulders=0, trenches=0, objects=False, calm_radius_m=1000.0, calm_slope=[0.1, 0.0], dock_xy_m=[0.0, 0.0], dock_floor_depth_m=10.0, center_m=[0.0, 0.0]))
    assert abs(t.height(0.0, 0.0) - 10.0) < 0.05 and abs(t.height(10.0, 3.0) - 11.0) < 0.05           # a plane sloping 0.1 m per m to the north
    xs = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert t.height(xs, xs).shape == (2, 2)                                                            # arrays in, arrays out
    n = t.normal(0.0, 0.0)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9 and n[2] < 0 and abs(n[0] - 0.0995) < 0.01              # unit, pointing up (NED -z), tilted by the slope
    assert abs(t.height(1e6, 0.0) - t.height(59.0, 0.0)) < 0.2                                         # clamped at the edge, no crash


def test_objects_stand_on_the_bottom_and_have_the_brightest_backscatter():
    flat = dict(SMALL, relief_m=0.0, ripples_m=0.0, boulders=0, trenches=0, calm_radius_m=1000.0, calm_slope=[0.0, 0.0], dock_xy_m=[0.0, 0.0], dock_floor_depth_m=10.0, center_m=[0.0, 0.0])
    t = Terrain(dict(flat, objects=True, object_list=[{"kind": "box", "xy": [0.0, 8.0], "size": [2.0, 2.0, 1.5]}, {"kind": "pipe", "from": [-30.0, -10.0], "to": [30.0, -10.0], "radius_m": 0.4}]))
    assert abs(t.height(0.0, 8.0) - 8.5) < 0.05                          # the box top is 1.5 m shallower than the floor
    assert abs(t.height(0.0, 20.0) - 10.0) < 0.05
    assert t.height(0.0, -10.0) < 9.4                                    # the pipe
    assert t.reflectivity_db(0.0, 8.0) > t.reflectivity_db(0.0, 20.0) + 10.0   # man-made objects ring louder than sand
    t0 = Terrain(dict(flat, objects=False))
    assert abs(t0.height(0.0, 8.0) - 10.0) < 0.05                        # negative control: without objects the floor is flat there
