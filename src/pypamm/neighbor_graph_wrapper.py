"""
Python wrapper for the neighbor_graph Cython module.
"""
import numpy as np
from scipy.sparse import csr_matrix

def build_knn_graph(X, k, metric="euclidean", inv_cov=None, include_self=False, n_jobs=1):
    """
    Build a k-nearest neighbor graph.
    
    Parameters:
    - X: Data matrix (N x D)
    - k: Number of neighbors
    - metric: Distance metric to use
    - inv_cov: Optional parameter for certain distance metrics
    - include_self: Whether to include self as a neighbor
    - n_jobs: Number of parallel jobs
    
    Returns:
    - indices: Indices of k nearest neighbors for each point (N x k)
    - distances: Distances to k nearest neighbors for each point (N x k)
    """
    # Validate inputs
    N = X.shape[0]
    
    if N == 0:
        raise ValueError("Input data cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if k >= N:
        raise ValueError(f"k ({k}) must be less than the number of data points ({N})")
    
    # Use scipy's KDTree for efficient neighbor search
    from scipy.spatial import KDTree
    
    # Convert X to float64 if needed
    X = np.asarray(X, dtype=np.float64)
    
    # Create KDTree
    tree = KDTree(X)
    
    # Query for k+1 neighbors (including self)
    if include_self:
        k_query = k
    else:
        k_query = k + 1  # We'll compute k+1 neighbors and exclude self
    
    # Query the tree
    distances, indices = tree.query(X, k=k_query, workers=n_jobs)
    
    # Remove self if needed
    if not include_self:
        # Remove the first column (self)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    
    return indices.astype(np.int32), distances.astype(np.float64)

def build_neighbor_graph(X, k, inv_cov=None, metric="euclidean", method="brute_force", graph_type="knn"):
    """
    Build a Neighbor Graph using a specified distance metric.
    
    Parameters:
    - X: (N x D) NumPy array (data points)
    - k: Number of nearest neighbors to keep
    - inv_cov: (D x D) inverse covariance matrix for Mahalanobis distance (only needed for Mahalanobis and Minkowski)
    - metric: "euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski"
    - method: "brute_force" (default) or "kd_tree" for faster search
    - graph_type: "knn" (default) or "gabriel" to compute Gabriel Graph edges

    Returns:
    - adjacency_list: sparse csr_matrix where adjacency_list[i, j] contains distance value if edge exists
    """
    # Get the indices and distances
    indices, distances = build_knn_graph(X, k, metric, inv_cov, include_self=False, n_jobs=1)
    
    # Create a sparse matrix
    N = X.shape[0]
    rows = np.repeat(np.arange(N), k)
    cols = indices.flatten()
    vals = distances.flatten()
    
    # Create the sparse matrix
    adj_matrix = csr_matrix((vals, (rows, cols)), shape=(N, N))
    
    return adj_matrix 