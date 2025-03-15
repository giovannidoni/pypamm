"""
Wrapper functions for the clustering utilities module.
"""

import numpy as np
from pypamm.clustering.utils import merge_clusters as _merge_clusters

def merge_clusters(X, cluster_labels, probabilities, covariance_matrices, threshold=0.8):
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
    return _merge_clusters(X, probabilities, cluster_labels, covariance_matrices, threshold) 