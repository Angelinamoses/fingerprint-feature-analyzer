"""Regression tests for the bugs found in the audit. Run: python -m pytest tests"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.minutiae import _crossing_number, crossing_number_map
from processing.pattern import compute_poincare_index
from processing.orientation import estimate_orientation


def _grid(theta_fn, n=9):
    ys, xs = np.mgrid[0:n, 0:n]
    z = (xs - n // 2) + 1j * (ys - n // 2)
    return theta_fn(z).astype(np.float32), np.ones((n, n), bool)


def test_poincare_core_is_plus_half():
    th, v = _grid(lambda z: 0.5 * np.angle(z))
    p = compute_poincare_index(th, v, radius=1)
    assert abs(p[3, 3] - 0.5) < 1e-4


def test_poincare_delta_is_minus_half():
    th, v = _grid(lambda z: -0.5 * np.angle(z))
    p = compute_poincare_index(th, v, radius=1)
    assert abs(p[3, 3] + 0.5) < 1e-4


def test_poincare_flat_is_zero():
    th, v = _grid(lambda z: np.full(z.shape, 0.3))
    assert abs(compute_poincare_index(th, v)[3, 3]) < 1e-6


def test_poincare_centre_block_may_be_invalid():
    th, v = _grid(lambda z: 0.5 * np.angle(z))
    v[4, 4] = False
    assert np.isfinite(compute_poincare_index(th, v)[3, 3])


def _true_cn(nb):
    p = nb.flatten()
    ring = [p[0], p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0]]
    return sum(abs(int(ring[i + 1]) - int(ring[i])) for i in range(8)) // 2


def test_crossing_number_all_configs():
    for bits in itertools.product([0, 1], repeat=8):
        nb = np.zeros(9, np.uint8)
        for b, i in zip(bits, [0, 1, 2, 3, 5, 6, 7, 8]):
            nb[i] = b
        nb[4] = 1
        nb = nb.reshape(3, 3)
        assert _crossing_number(nb) == _true_cn(nb)
        assert crossing_number_map(np.pad(nb, 1).astype(bool))[2, 2] == _true_cn(nb)


def test_orientation_is_ridge_direction_not_gradient():
    h = w = 128
    ys, xs = np.mgrid[0:h, 0:w]
    ridge_angle = np.radians(30)             # ridges run at 30 deg
    normal = -xs * np.sin(ridge_angle) + ys * np.cos(ridge_angle)
    img = (127 + 100 * np.sin(2 * np.pi * normal / 10)).astype(np.uint8)
    ori, coh = estimate_orientation(img)
    est = float(np.median(ori[32:-32, 32:-32]))
    diff = abs(np.angle(np.exp(2j * (est - ridge_angle)))) / 2
    assert np.degrees(diff) < 3


if __name__ == "__main__":      # runs without pytest
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failed += 1
                print("FAIL", name)
    sys.exit(failed)
