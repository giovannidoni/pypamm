"""
Wrapper functions for the cluster_covariance module.
"""

import numpy as np
from pypamm.clustering.cluster_covariance import compute_cluster_covariance as _compute_cluster_covariance

def compute_cluster_covariance(X, cluster_labels, regularization=None):
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
    cov_matrices = _compute_cluster_covariance(X, cluster_labels)
    
    # Apply regularization if requested
    if regularization is not None:
        if not isinstance(regularization, (int, float)):
            raise ValueError("regularization must be a number")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        
        n_clusters, n_features, _ = cov_matrices.shape
        for i in range(n_clusters):
            # Add regularization to the diagonal
            np.fill_diagonal(cov_matrices[i], 
                             np.diag(cov_matrices[i]) + regularization)
    
    return cov_matrices 