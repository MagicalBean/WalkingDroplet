# SUBPLANE fit and subtract a 2d plane from a matrix representing elevation

import numpy as np
from numpy.linalg import solve

def subplane(h):
    """Returns the normalized height map.

    Args:
        h: original height field

    Returns:
        Height field with fitted plane subtracted from it
    """
    rows, cols = h.shape
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    A = np.column_stack((np.ones(rows * cols), r.ravel(), c.ravel()))

    fitc, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)

    h_sub = h - (A @ fitc).reshape(h.shape)

    return h_sub