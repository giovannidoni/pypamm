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
    X = np.ascontiguousarray(X.astype(np.float64))

    # Call the Cython implementation
    return _select_grid_points(X, ngrid, metric, inv_cov)


def compute_voronoi(
    X: NDArray[np.float64],
    wj: NDArray[np.float64],
    Y: NDArray[np.float64],
    idxgrid: NDArray[np.int32],
    metric: str = "euclidean",
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64], NDArray[np.int32]]:
    """
    Assign each sample in X to the closest grid point in Y (Voronoi assignment).

    Parameters:
    - X: Sample matrix (N x D)
    - wj: Weights for each sample (N,)
    - Y: Grid points (ngrid x D)
    - idxgrid: Indices of selected grid points (ngrid,)
    - metric: Distance metric (e.g. "euclidean")

    Returns:
    - iminij: Attribution of each sample to a grid point (N,)
    - ni: Number of samples per Voronoi cell (ngrid,)
    - wi: Sum of weights per Voronoi cell (ngrid,)
    - ineigh: Closest sample index to each grid point (ngrid,)
    """
    from pypamm.voronoi import compute_voronoi as _compute_voronoi

    X = np.ascontiguousarray(X, dtype=np.float64)
    wj = np.ascontiguousarray(wj, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    idxgrid = np.ascontiguousarray(idxgrid, dtype=np.int32)

    return _compute_voronoi(X, wj, Y, idxgrid, metric=metric)
