"""
Wrapper functions for the cluster utilities module.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Import the Cython implementation directly
from pypamm.clustering.cluster_utils import (
    compute_cluster_covariance as _compute_cluster_covariance_cython,
)
from pypamm.clustering.cluster_utils import (
    merge_clusters as _merge_clusters_cython,
)
from pypamm.clustering.cluster_utils import (
    reindex_clusters as _reindex_clusters_cython,
)

# Export all functions
__all__ = [
    "compute_cluster_covariance",
    "merge_clusters",
    "reindex_clusters",
]


def compute_cluster_covariance(
    X: ArrayLike, cluster_labels: ArrayLike, regularization: float | None = None
) -> NDArray[np.float64]:
    """
    Computes covariance matrices for each cluster.

    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data points.

    cluster_labels : array-like, shape (n_samples,)
        Cluster assignments for each data point. Should be integers starting from 0.

    regularization : float, optional
        If provided, adds a regularization term to the diagonal of each covariance matrix.
        This can help ensure numerical stability and prevent singular matrices.

    Returns:
    -------
    cov_matrices : array, shape (n_clusters, n_features, n_features)
        Covariance matrices for each cluster.
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.asarray(X, dtype=np.float64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

    # Validate inputs
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if cluster_labels.ndim != 1:
        raise ValueError("cluster_labels must be a 1D array")
    if len(cluster_labels) != X.shape[0]:
        raise ValueError("Length of cluster_labels must match the number of samples in X")

    # Compute covariance matrices
    cov_matrices = _compute_cluster_covariance_cython(X, cluster_labels)

    # Apply regularization if requested
    if regularization is not None:
        if not isinstance(regularization, int | float):
            raise ValueError("regularization must be a number")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")

        n_clusters, n_features, _ = cov_matrices.shape
        for i in range(n_clusters):
            # Add regularization to the diagonal
            np.fill_diagonal(cov_matrices[i], np.diag(cov_matrices[i]) + regularization)

    return cov_matrices


def merge_clusters(
    X: ArrayLike,
    cluster_labels: ArrayLike,
    probabilities: ArrayLike,
    covariance_matrices: ArrayLike,
    threshold: float = 0.8,
) -> NDArray[np.int32]:
    """
    Merges weak clusters based on probability and adjacency.

    This function identifies clusters with low probability mass and merges them
    into nearby clusters based on Mahalanobis distance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data points.

    cluster_labels : array-like, shape (n_samples,)
        Cluster assignments for each data point. Should be integers starting from 0.

    probabilities : array-like, shape (n_samples,)
        Probabilities associated with each data point, typically from a density estimation.

    covariance_matrices : array-like, shape (n_clusters, n_features, n_features)
        Covariance matrices for each cluster.

    threshold : float, default=0.8
        Probability threshold below which a cluster should be merged.
        Higher values result in more aggressive merging.

    Returns
    -------
    new_labels : array, shape (n_samples,)
        Updated cluster assignments after merging weak clusters.

    Notes
    -----
    This function is useful for post-processing clustering results to eliminate
    small or weak clusters that may be artifacts or noise.
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.asarray(X, dtype=np.float64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    covariance_matrices = np.asarray(covariance_matrices, dtype=np.float64)

    # Validate inputs
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if cluster_labels.ndim != 1:
        raise ValueError("cluster_labels must be a 1D array")
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a 1D array")
    if covariance_matrices.ndim != 3:
        raise ValueError("covariance_matrices must be a 3D array")
    if len(cluster_labels) != X.shape[0]:
        raise ValueError("Length of cluster_labels must match the number of samples in X")
    if len(probabilities) != X.shape[0]:
        raise ValueError("Length of probabilities must match the number of samples in X")
    if covariance_matrices.shape[0] <= np.max(cluster_labels):
        raise ValueError("Number of covariance matrices must be at least the number of clusters")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    # Call the Cython implementation
    return _merge_clusters_cython(X, probabilities, cluster_labels, covariance_matrices, threshold)


def reindex_clusters(cluster_labels: ArrayLike) -> NDArray[np.int32]:
    """
    Ensures cluster labels are contiguous (reindexes clusters).

    Parameters:
    ----------
    cluster_labels : array-like, shape (n_samples,)
        Cluster assignments for each data point. Should be integers.

    Returns:
    -------
    new_labels : array, shape (n_samples,)
        Updated cluster labels with contiguous numbering.
    """
    # Convert input to numpy array if it isn't already
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

    # Call the Cython implementation
    return _reindex_clusters_cython(cluster_labels)
