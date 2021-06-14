

import numpy as np

from . import setup_greedy, setup_random, opt_maxvol, opt_dmetric


def test_setup(rng):
    ncol = 5
    for _ in range(100):
        rows = rng.normal(0, 1, (20, ncol))
        perm1 = setup_greedy(rows)
        assert len(perm1) == len(set(perm1))
        square1 = rows[perm1[:ncol]]
        perm2 = setup_random(rows, rng)
        assert len(perm2) == len(set(perm2))
        square2 = rows[perm2[:ncol]]
        assert abs(np.linalg.det(square1)) > abs(np.linalg.det(square2))


def test_opt_maxvol(rng):
    ncol = 5
    for _ in range(100):
        rows = rng.normal(0, 1, (15, ncol))
        perm = opt_maxvol(rows)
        assert len(perm) == len(set(perm))
        square_opt = rows[perm[:ncol]]
        square = rows[:ncol]
        assert abs(np.linalg.det(square_opt)) > abs(np.linalg.det(square))


def test_opt_dmetric(rng):
    ncol = 5
    nrow = 9
    for _ in range(100):
        rows = rng.normal(0, 1, (25, ncol))
        perm = opt_dmetric(rows, nrow)
        assert len(perm) == len(set(perm))
        det1 = np.linalg.det(np.dot(rows[perm[:nrow]].T, rows[perm[:nrow]]))
        det2 = np.linalg.det(np.dot(rows[:nrow].T, rows[:nrow]))
        assert abs(det1) > abs(det2)
