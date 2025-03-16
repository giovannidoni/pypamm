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
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
    neighbor_graph: spmatrix | None = None,
) -> NDArray[np.int32]:
    """
    Vanilla Quick-Shift clustering algorithm based on density gradient ascent.

    This implementation takes pre-computed probability values and does not use KDE internally.

    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points.
    prob : array-like, shape (n_samples,), optional
        Probability estimates for each point. If None, uniform probabilities are used.
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

    Returns:
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point.
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)

    # If prob is None, use uniform probabilities
    if prob is None:
        prob = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
    else:
        prob = np.asarray(prob, dtype=np.float64)

    # If neighbor_graph is provided, use it for clustering
    if neighbor_graph is not None:
        # Implement neighbor graph-based quick shift
        return _neighbor_graph_quick_shift(X, prob, neighbor_graph, lambda_qs)
    else:
        # Call the Cython implementation with the correct parameter order
        # Extract just the labels from the tuple returned by _quick_shift_clustering
        labels, _ = _quick_shift_clustering(X, prob, ngrid, None, metric, lambda_qs, max_dist)
        return labels


def _neighbor_graph_quick_shift(
    X: NDArray[np.float64],
    prob: NDArray[np.float64],
    neighbor_graph: spmatrix,
    lambda_qs: float = 1.0,
) -> NDArray[np.int32]:
    """
    Quick-Shift clustering using a pre-computed neighbor graph.

    Parameters:
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data points.
    prob : ndarray of shape (n_samples,)
        Probability estimates for each point.
    neighbor_graph : scipy.sparse matrix
        Pre-computed neighbor graph. Should be a sparse matrix where non-zero entries
        indicate connected points.
    lambda_qs : float, default=1.0
        Scaling factor for density-based traversal.

    Returns:
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point.
    """
    n_samples = X.shape[0]

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


def quick_shift_kde(
    X: ArrayLike,
    bandwidth: float,
    ngrid: int = 100,
    metric: str = "euclidean",
    lambda_qs: float = 1.0,
    max_dist: float = np.inf,
    neighbor_graph: spmatrix | None = None,
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
        Bandwidth parameter for KDE.
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
    prob = compute_kde(X, X, bandwidth)

    # Call the vanilla quick_shift with pre-computed probabilities
    return quick_shift(X, prob, ngrid, metric, lambda_qs, max_dist, neighbor_graph)
