"""
Wrapper functions for the quick_shift module.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypamm.quick_shift import quick_shift_clustering as _quick_shift_clustering


def quick_shift(
    X: ArrayLike,
    prob: ArrayLike | None = None,
    ngrid: int = 100,
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Vanilla Quick-Shift clustering algorithm based on density gradient ascent.

    This implementation takes pre-computed probability values and does not use KDE internally.

    Parameters:
    - X: (N x D) NumPy array (data points)
    - prob: (N,) NumPy array of probability estimates for each point. If None, uniform probabilities are used.
    - ngrid: Number of grid points
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski")
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold for connecting points (default: infinity)

    Returns:
    - idxroot: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)

    # If prob is None, use uniform probabilities
    if prob is None:
        prob = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
    else:
        prob = np.asarray(prob, dtype=np.float64)

    # Call the Cython implementation
    return _quick_shift_clustering(X, prob, ngrid, metric, lambda_qs, max_dist)


def quick_shift_kde(
    X: ArrayLike,
    bandwidth: float,
    ngrid: int = 100,
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    KDE-enhanced Quick-Shift clustering algorithm.

    This implementation computes probability densities using Kernel Density Estimation (KDE)
    before applying the Quick-Shift algorithm.

    Parameters:
    - X: (N x D) NumPy array (data points)
    - bandwidth: Bandwidth parameter for KDE
    - ngrid: Number of grid points
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski")
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold for connecting points (default: infinity)

    Returns:
    - idxroot: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)

    # Import KDE function
    from pypamm.density import compute_kde

    # Compute probability densities using KDE
    prob = compute_kde(X, X, bandwidth)

    # Call the vanilla quick_shift with pre-computed probabilities
    return quick_shift(X, prob, ngrid, metric, lambda_qs, max_dist)
