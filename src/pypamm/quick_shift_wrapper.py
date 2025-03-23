"""
Wrapper functions for the quick_shift module.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import spmatrix

from pypamm.quick_shift import quick_shift_clustering as _quick_shift_clustering


def quick_shift(
    X: ArrayLike,
    prob: ArrayLike | None = None,
    ngrid: int = 100,
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
    neighbor_graph: spmatrix | None = None,
    metric: str = "euclidean",
    k: int = 2,
    inv_cov: NDArray[np.float64] | None = None,
) -> NDArray[np.int32]:
    """
    Quick-Shift clustering algorithm based on density gradient ascent.
    This implementation can work with either pairwise distances or a pre-computed
    neighbor graph, automatically choosing the most efficient approach.
    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points.
    prob : array-like, shape (n_samples,), optional
        Probability estimates for each point. If None, uniform probabilities are used.
    ngrid : int, default=100
        Number of grid points (only used when neighbor_graph is None).
    metric : str, default="euclidean"
        Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski").
        Only used when neighbor_graph is None.
    lambda_qs : float, default=1.0
        Scaling factor for density-based traversal.
    max_dist : float, default=np.inf
        Maximum distance threshold for connecting points.
        Only used when neighbor_graph is None.
    neighbor_graph : scipy.sparse matrix, optional
        Pre-computed neighbor graph. If provided, this will be used instead of computing
        distances between all points, which is more efficient for large datasets.
    k : int, default=2
        Exponent for the Minkowski distance.
        Only used when metric="minkowski".
    inv_cov : array-like, shape (n_features, n_features), optional
        Inverse covariance matrix for Mahalanobis distance.
        Only used when metric="mahalanobis".

    Returns:
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point.
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    # If prob is None, use uniform probabilities with small random variations
    if prob is None:
        prob = np.ones(n_samples, dtype=np.float64) / n_samples
    else:
        prob = np.asarray(prob, dtype=np.float64)

    # Choose the appropriate implementation based on whether a neighbor graph is provided
    if neighbor_graph is not None:
        # Neighbor graph-based implementation (more efficient for large datasets)

        # Initialize parent array (each point starts as its own parent)
        parents = np.arange(n_samples, dtype=np.int32)

        # For each point, find the neighbor with highest density
        for i in range(n_samples):
            # Get neighbors of point i
            neighbors = neighbor_graph.getrow(i).indices

            if len(neighbors) > 0:
                # Find neighbor with highest density
                neighbor_probs = prob[neighbors]

                # Only consider neighbors with higher density
                higher_density_mask = neighbor_probs > prob[i] * lambda_qs

                if np.any(higher_density_mask):
                    # Get indices of neighbors with higher density
                    higher_density_neighbors = neighbors[higher_density_mask]
                    higher_density_probs = neighbor_probs[higher_density_mask]

                    # Find the neighbor with highest density
                    best_neighbor = higher_density_neighbors[np.argmax(higher_density_probs)]

                    # Set parent to the best neighbor
                    parents[i] = best_neighbor

        # Propagate labels to find cluster roots
        labels = np.zeros(n_samples, dtype=np.int32)
        cluster_id = 0

        for i in range(n_samples):
            if parents[i] == i:  # This is a root node
                # Assign a new cluster ID
                labels[i] = cluster_id
                cluster_id += 1
            else:
                # Follow the path to the root
                current = i
                path = [current]

                while parents[current] != current:
                    current = parents[current]
                    path.append(current)

                    # Avoid infinite loops
                    if len(path) > n_samples:
                        break

                # Assign the root's cluster ID to all points in the path
                root = path[-1]

                # If the root doesn't have a label yet, assign one
                if labels[root] == 0 and root != 0:
                    labels[root] = cluster_id
                    cluster_id += 1

                # Assign the root's label to this point
                labels[i] = labels[root]

        return labels
    else:
        # Traditional implementation using the Cython code
        # Extract just the labels from the tuple returned by _quick_shift_clustering
        labels, _ = _quick_shift_clustering(X, prob, ngrid, None, lambda_qs, max_dist, metric, k, inv_cov)
        return labels


def quick_shift_kde(
    X: ArrayLike,
    bandwidth: float,
    ngrid: int = 100,
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
    neighbor_graph: spmatrix | None = None,
    adaptive: bool = True,
    k: int = 2,
    inv_cov: NDArray[np.float64] | None = None,
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
    metric : str, default="euclidean"
        Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski").
    k : int, default=2
        Exponent for the Minkowski distance.
        Only used when metric="minkowski".
    inv_cov : array-like, shape (n_features, n_features), optional
        Inverse covariance matrix for Mahalanobis distance.
        Only used when metric="mahalanobis".

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
    return _quick_shift_clustering(X, prob, ngrid, neighbor_graph, lambda_qs, max_dist, metric, k, inv_cov)
