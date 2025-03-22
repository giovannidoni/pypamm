"""
Wrapper functions for the quick_shift module.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import spmatrix

from pypamm.quick_shift import quick_shift_clustering as _quick_shift_clustering


def quick_shift_kde(
    X: ArrayLike,
    bandwidth: float,
    ngrid: int = 100,
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
    neighbor_graph: spmatrix | None = None,
    adaptive: bool = True,
) -> NDArray[np.int32]:
    """
    KDE-enhanced Quick-Shift clustering algorithm.

    This implementation computes probability densities using Kernel Density Estimation (KDE)
    before applying the Quick-Shift algorithm.

    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points.
    bandwidth : float
        Bandwidth parameter for KDE. If adaptive=True, this is used as the alpha parameter
        for adaptive bandwidth. Otherwise, it's used as a fixed bandwidth.
    ngrid : int, default=100
        Number of grid points.
    metric : str, default="euclidean"
        Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski").
    lambda_qs : float, default=1.0
        Scaling factor for density-based traversal.
    max_dist : float, default=np.inf
        Maximum distance threshold for connecting points.
    neighbor_graph : scipy.sparse matrix, optional
        Pre-computed neighbor graph. If provided, this will be used instead of computing
        distances between all points.
    adaptive : bool, default=True
        Whether to use adaptive bandwidth for KDE. If True, the bandwidth parameter is used
        as the alpha parameter for adaptive bandwidth. If False, it's used as a fixed bandwidth.

    Returns:
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point.
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)

    # Import KDE function
    from pypamm.density.kde import compute_kde

    # Compute probability densities using KDE
    if adaptive:
        # Use bandwidth as alpha parameter for adaptive bandwidth
        prob = compute_kde(X, X, alpha=bandwidth, adaptive=True)
    else:
        # Use bandwidth as fixed bandwidth
        prob = compute_kde(X, X, constant_bandwidth=bandwidth, adaptive=False)

    # Call the unified quick_shift with pre-computed probabilities
    return _quick_shift_clustering(X, prob, ngrid, neighbor_graph, metric, lambda_qs, max_dist)
