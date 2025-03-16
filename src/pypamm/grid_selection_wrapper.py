"""
Python wrapper for the grid_selection Cython module.
"""

import numpy as np
from numpy.typing import NDArray


def select_grid_points(
    X: NDArray[np.float64], ngrid: int, metric: str = "euclidean", inv_cov: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """
    Select grid points from a dataset.

    Parameters:
    - X: Data matrix (N x D)
    - ngrid: Number of grid points to select
    - metric: Distance metric to use
    - inv_cov: Optional parameter for certain distance metrics

    Returns:
    - idxgrid: Indices of selected grid points
    - Y: Selected grid points
    """
    # Import the Cython implementation
    from pypamm.grid_selection import select_grid_points as _select_grid_points

    # Convert X to float64 if needed
    X = np.asarray(X, dtype=np.float64)

    # Call the Cython implementation
    return _select_grid_points(X, ngrid, metric, inv_cov)
