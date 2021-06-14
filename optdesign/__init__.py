

import numpy as np


__all__ = ["setup_greedy", "setup_random", "opt_maxvol", "opt_dmetric"]


def _swap(perm, i, j):
    """Swap two indexes in a permutation."""
    perm[i], perm[j] = perm[j], perm[i]


def setup_greedy(rows):
    """Construct a square matrix A, approx maximizing abs(det(A)).

    Parameters
    ----------
    rows
        A matrix with candidate rows, shape ``(n, m)`` with ``n`` > ``m``.

    Returns
    -------
    perm
        The permutation to be applied to the rows. The first ``m`` integers are used to
        construct the square matrix.

    """
    if rows.ndim != 2 or rows.shape[0] < rows.shape[1]:
        raise TypeError("Argument rows must not have less rows than columns.")
    perm = list(range(rows.shape[0]))
    # Select first row with the largest norm.
    row_norms = np.linalg.norm(rows, axis=1)
    _swap(perm, 0, np.argmax(row_norms))
    # Iteratively select columns that will maximize the absolute value of
    # the product of non-zero singular values of the (not yet) square matrix.
    for irow in range(1, rows.shape[1]):
        work = rows[perm[:irow]]
        U, S, Vt = np.linalg.svd(work, full_matrices=False)
        candidates = rows[perm[irow:]]
        factors = np.linalg.norm(np.dot(np.dot(candidates, Vt.T), Vt) - candidates, axis=1)
        assert factors.ndim == 1
        assert factors.shape[0] == candidates.shape[0]
        _swap(perm, irow, factors.argmax() + irow)
    return perm


def setup_random(rows, rng=None):
    """Return a random permutation of the rows.

    Parameters
    ----------
    rows
        A matrix with candidate rows, shape ``(n, m)`` with ``n`` > ``m``.

    Returns
    -------
    perm
        The permutation to be applied to the rows.

    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.permutation(rows.shape[0])


def opt_maxvol(rows, maxiter=None, threshold=1e-15):
    """Maximize abs det of the square matrix formed by the first rows.

    Parameters
    ----------
    rows
        A matrix with candidate rows, shape ``(n, m)`` with ``n`` > ``m``.
    maxiter
        Maximum number if iterations. If not given, a sensible choice is made
        automatically.
    threshold
        Stop early when the imptovement to the determinant is lower than this threshold.

    Returns
    -------
    perm
        The permutation to be applied to the rows. The first ``m`` integers are used to
        construct the square matrix.

    """
    if rows.ndim != 2 or rows.shape[0] < rows.shape[1]:
        raise TypeError("Argument rows must not have less rows than columns.")
    if maxiter is None:
        maxiter = rows.shape[1] * 2
    perm = list(range(rows.shape[0]))
    ncol = rows.shape[1]
    for _ in range(maxiter):
        square = rows[perm[:ncol]]
        candidates = rows[perm[ncol:]]
        square_inv = np.linalg.inv(square)
        factors = abs(np.dot(candidates, square_inv))
        inew, iold = np.unravel_index(np.argmax(factors, axis=None), factors.shape)
        if factors[inew, iold] - 1 < threshold:
            break
        _swap(perm, inew + ncol, iold)
    return perm


def opt_dmetric(rows, nrow, maxiter=None, threshold=1e-15):
    """Simple refinement of d-optimality

    Parameters
    ----------
    rows
        A matrix with candidate rows, shape ``(n, m)`` with ``n`` > ``m``.
    nrow
        The number of rows of interest.
    maxiter
        Maximum number if iterations. If not given, a sensible choice is made
        automatically.
    threshold
        Stop early when the imptovement to the determinant is lower than this threshold.

    Returns
    -------
    perm
        The permutation to be applied to the rows. The first ``nrow`` integers are used to
        construct the square matrix.

    """
    if rows.ndim != 2 or rows.shape[0] < rows.shape[1]:
        raise TypeError("Argument rows must not have less rows than columns.")
    if nrow < rows.shape[1]:
        raise TypeError("nrow must be at least equal to the number of columns.")
    if maxiter is None:
        maxiter = nrow * 2

    def get_sqnorms(selected, candidates):
        U, s, Vt = np.linalg.svd(selected)
        tf_candidates = np.dot(candidates, Vt.T) / s
        sqnorms = np.einsum("ij,ij->i", tf_candidates, tf_candidates)
        return sqnorms

    perm = list(range(rows.shape[0]))
    for _ in range(maxiter):
        # Add one row
        selected = rows[perm[:nrow]]
        candidates = rows[perm[nrow:]]
        sqnorms_add = get_sqnorms(selected, candidates)
        iadd = sqnorms_add.argmax()
        _swap(perm, iadd + nrow, nrow)
        # Delete one row
        selected = rows[perm[:nrow + 1]]
        sqnorms_del = get_sqnorms(selected, selected)
        idel = sqnorms_del.argmin()
        # Check convergence
        factor_minus_one = sqnorms_add[iadd] - sqnorms_del[idel] - \
            sqnorms_del[idel] * sqnorms_add[iadd]
        if factor_minus_one < threshold:
            break
        _swap(perm, idel, nrow)
    return perm
