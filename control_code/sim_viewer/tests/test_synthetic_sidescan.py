"""The side-scan physics (synthetic_sidescan.py) against analytic cases: swath geometry, nadir gap, shadows, gain, absorption, port/starboard symmetry, mounting, ping rate."""
import math

import numpy as np
import pytest

from synthetic_sidescan import SideScanConfig, SideScanSim, flat_swath_geometry
from terrain import Terrain

FLAT = {"size_m": 120.0, "dx_m": 0.25, "boulders": 0, "trenches": 0, "relief_m": 0.0, "ripples_m": 0.0, "center_m": [0.0, 0.0], "dock_xy_m": [0.0, 0.0], "calm_radius_m": 1000.0,
        "dock_floor_depth_m": 10.0, "calm_slope": [0.0, 0.0], "objects": False}
POS, EUL = [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]           # 7 m above a flat floor at 10 m, heading north: starboard looks east


def ping(sim, side="starboard", pos=POS, eul=EUL, n=6):
    for _ in range(n):
        p = sim.ping(pos, eul, side)                   # a few pings: auto gain settles
    return p


def inten(p):
    return p.intensity.astype(float) / 65535.0


def rng_of(p, i):
    return (i + 0.5) * p.range_m / len(p.intensity)


@pytest.fixture(scope="module")
def flat():
    return Terrain(FLAT)


def test_flat_bottom_swath_matches_the_analytic_geometry(flat):
    sim = SideScanSim(flat, SideScanConfig(gain_index=3, range_m=30.0))
    g = flat_swath_geometry(7.0)
    p = ping(sim)
    v = inten(p)
    echo = np.nonzero(v > 0.15)[0]
    assert abs(rng_of(p, echo[0]) - g["slant_near_m"]) < 0.6, (rng_of(p, echo[0]), g)          # first echo = nearest edge of the 50 deg fan (5 deg from the nadir)
    assert v[:int((g["slant_near_m"] - 0.5) / 30 * 600)].max() < 0.1                              # the nadir gap below is empty (noise only)
    assert rng_of(p, echo[-1]) < g["slant_far_m"] + 4.0                                           # echoes die out beyond the fan (the pattern tapers it a little past -3 dB)


def test_port_and_starboard_are_mirror_images_over_a_flat_floor(flat):
    sim = SideScanSim(flat, SideScanConfig(gain_index=3, looks=50.0))
    a, b = inten(ping(sim, "port")), inten(ping(sim, "starboard"))
    lo, hi = int(7.5 / 30 * 600), int(13 / 30 * 600)
    assert abs(a[lo:hi].mean() - b[lo:hi].mean()) < 0.06


def test_a_boulder_casts_a_shadow_of_the_analytic_length_and_no_boulder_no_shadow():
    box = {"kind": "box", "xy": [0.0, 8.0], "size": [8.0, 2.0, 1.5]}
    with_box = SideScanSim(Terrain(dict(FLAT, objects=True, object_list=[box])), SideScanConfig(gain_index=3, range_m=30.0))
    without = SideScanSim(Terrain(FLAT), SideScanConfig(gain_index=3, range_m=30.0))
    p1, p0 = ping(with_box), ping(without)
    H, h, g_top = 7.0, 1.5, 9.0
    r0, r1 = math.hypot(g_top, H - h), math.hypot(g_top * H / (H - h), H)                         # slant range of the top edge and where the shadow ends (10.55 .. 13.42 m)
    a, b = int((r0 + 0.3) / 30 * 600), int((r1 - 0.3) / 30 * 600)
    assert inten(p1)[a:b].mean() < 0.03                                                          # dark
    assert inten(p0)[a:b].mean() > 0.25                                                          # negative control: the same bins are bright without the box
    first_dark = np.nonzero(inten(p1)[int(9.5 / 30 * 600):] < 0.03)[0][0] + int(9.5 / 30 * 600)
    assert abs(rng_of(p1, first_dark) - r0) < 0.4                                                # the shadow starts at the top edge of the box


def test_gain_absorption_and_bins_and_range_commands(flat):
    s = SideScanSim(flat, SideScanConfig(gain_index=0))
    lo = ping(s)
    s.set_gain(7)
    hi = ping(s)
    band = slice(int(8 / 30 * 600), int(12 / 30 * 600))
    assert inten(hi)[band].mean() > inten(lo)[band].mean() + 0.2                                  # 42 dB more gain
    clear = SideScanSim(flat, SideScanConfig(gain_index=3, absorption_db_per_m=0.0, tvg=False, looks=50.0))
    murky = SideScanSim(flat, SideScanConfig(gain_index=3, absorption_db_per_m=0.5, tvg=False, looks=50.0))
    far = slice(int(11 / 30 * 600), int(13 / 30 * 600))
    assert inten(ping(clear))[far].mean() > inten(ping(murky))[far].mean() + 0.02                # more absorption, weaker far echo (negative control for the physics)
    s.set_bins(900)
    s.set_range(15.0)
    p = ping(s)
    assert len(p.intensity) == 900 and p.range_m == 15.0 and p.intensity.dtype == np.uint16
    s.set_range(1000.0)
    assert s.cfg.range_m == 150.0                                                                  # clamped to the sonar's maximum


def test_ping_rate_follows_the_two_way_travel_time():
    s = SideScanSim(Terrain(FLAT), SideScanConfig(range_m=15.0))
    assert s.ping_hz() == 20.0                                                                     # capped at the sonar's 20 Hz
    s.set_range(100.0)
    assert abs(s.ping_hz() - 0.9 * 1500 / 200.0) < 1e-9


def test_mount_angle_and_roll_move_the_swath(flat):
    steep = SideScanSim(flat, SideScanConfig(gain_index=3, mount_from_nadir_deg=60.0))
    shallow = SideScanSim(flat, SideScanConfig(gain_index=3, mount_from_nadir_deg=30.0))
    g60, g30 = flat_swath_geometry(7.0, steep.cfg), flat_swath_geometry(7.0, shallow.cfg)
    p60, p30 = ping(steep), ping(shallow)
    f60 = rng_of(p60, np.nonzero(inten(p60) > 0.15)[0][0])
    f30 = rng_of(p30, np.nonzero(inten(p30) > 0.15)[0][0])
    assert abs(f60 - g60["slant_near_m"]) < 1.0 and abs(f30 - g30["slant_near_m"]) < 1.0 and f60 > f30 + 0.5       # (the pattern taper lets a little echo in before the -3 dB edge)
    # rolling the vehicle 15 deg to starboard turns the starboard beam more downward (shorter reach) and the port beam more sideways (longer reach)
    lv = SideScanSim(flat, SideScanConfig(gain_index=3))
    ro = SideScanSim(flat, SideScanConfig(gain_index=3))
    roll = [math.radians(15.0), 0.0, 0.0]
    last = lambda p: np.nonzero(inten(p) > 0.15)[0][-1]
    assert last(ping(ro, "starboard", eul=roll)) < last(ping(lv, "starboard")) - 8
    assert last(ping(ro, "port", eul=roll)) > last(ping(lv, "port")) + 8


def test_altimeter_reads_the_altitude_with_small_noise(flat):
    s = SideScanSim(flat, SideScanConfig())
    vals = [s.altimeter(POS, EUL) for _ in range(50)]
    assert abs(np.mean(vals) - 7.0) < 0.1 and 0.0 < np.std(vals) < 0.2
    assert s.altimeter([0.0, 0.0, -80.0], EUL) is None                                              # beyond the altimeter's 60 m range
