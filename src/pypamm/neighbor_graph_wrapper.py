"""
Python wrapper for the neighbor_graph Cython module.
"""
from typing import Tuple, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import csr_matrix

def build_knn_graph(
    X: ArrayLike, 
    k: int, 
    metric: str = "euclidean", 
    inv_cov: Optional[NDArray[np.float64]] = None, 
    include_self: bool = False, 
    n_jobs: int = 1
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
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

def build_neighbor_graph(
    X: ArrayLike, 
    k: int, 
    inv_cov: Optional[NDArray[np.float64]] = None, 
    metric: str = "euclidean", 
    method: Literal["brute_force", "kd_tree"] = "brute_force", 
    graph_type: Literal["gabriel", "knn", "relative_neighborhood"] = "gabriel"
) -> csr_matrix:
    """
    Build a neighbor graph.
    
    Parameters:
    - X: Data matrix (N x D)
    - k: Number of neighbors to consider
    - inv_cov: Optional parameter for certain distance metrics
    - metric: Distance metric to use
    - method: Method to use for neighbor search
    - graph_type: Type of graph to build
    
    Returns:
    - graph: Sparse adjacency matrix representing the neighbor graph
    """
    # Validate inputs
    N = X.shape[0]
    D = X.shape[1]
    
    # Validate metric
    valid_metrics = ["euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski"]
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Valid options are: {valid_metrics}")
    
    # Validate inv_cov for special metrics
    if metric == "mahalanobis":
        if inv_cov is None:
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        if inv_cov.shape[0] != D or inv_cov.shape[1] != D:
            raise ValueError(f"inv_cov must be ({D},{D}) for Mahalanobis.")
    elif metric == "minkowski":
        if inv_cov is None:
            raise ValueError("Must supply a 1x1 array with exponent for Minkowski.")
        if inv_cov.shape[0] != 1 or inv_cov.shape[1] != 1:
            raise ValueError("For Minkowski distance, inv_cov must be a 1x1 array with param[0,0] = k.")
    
    # Get the indices and distances
    indices, distances = build_knn_graph(X, k, metric, inv_cov, include_self=False, n_jobs=1)
    
    # Create a list of lists of tuples (index, distance)
    adjacency_list = []
    
    for i in range(N):
        neighbors = []
        for j in range(k):
            neighbors.append((indices[i, j], distances[i, j]))
        adjacency_list.append(neighbors)
    
    # Import the Cython implementation
    from pypamm.neighbor_graph import build_neighbor_graph as _build_neighbor_graph
    
    # Call the Cython implementation
    indices, indptr, data = _build_neighbor_graph(X, k, inv_cov, metric, method, graph_type)
    
    # Create a sparse matrix
    return csr_matrix((data, indices, indptr), shape=(N, N)) 