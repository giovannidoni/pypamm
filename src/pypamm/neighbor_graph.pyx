# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, pow
from scipy.spatial import cKDTree  # Faster neighbor search
from scipy.sparse import csr_matrix  # Sparse storage for adjacency graph
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# Import distance functions from the distance_metrics module
from pypamm.distance_metrics cimport (
    dist_func_t,
    dist_euclidean,
    dist_manhattan,
    dist_chebyshev,
    dist_cosine,
    dist_mahalanobis,
    dist_minkowski,
    _get_distance_function
)
from pypamm.distance_metrics import get_distance_function

# Define a structure to hold a neighbor and its distance
ctypedef struct neighbor_t:
    int idx
    double dist

# Public API - this doesn't actually control what's exported in Cython
# The functions need to be properly defined and visible at the module level
__all__ = ['build_neighbor_graph', 'build_knn_graph', 'compute_knn_for_point']

# Function to build a k-nearest neighbor graph
cpdef tuple build_knn_graph(np.ndarray[np.float64_t, ndim=2] X, int k, str metric, 
                   object inv_cov, bint include_self, int n_jobs):
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
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    
    if N == 0:
        raise ValueError("Input data cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if k >= N:
        raise ValueError(f"k ({k}) must be less than the number of data points ({N})")
    
    cdef int effective_k = k
    cdef int i
    
    if include_self:
        effective_k = k
    else:
        effective_k = k + 1  # We'll compute k+1 neighbors and exclude self
    
    # Process inv_cov
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    if inv_cov is None:
        if metric == "mahalanobis":
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        elif metric == "minkowski":
            # Default to Euclidean distance (p=2) if not specified
            inv_cov_arr = np.zeros((1, 1), dtype=np.float64)
            inv_cov_arr[0, 0] = 2.0
        else:
            inv_cov_arr = np.zeros((1, 1), dtype=np.float64)
    else:
        inv_cov_arr = inv_cov
        
        # Validate inv_cov dimensions
        if metric == "mahalanobis":
            if inv_cov_arr.shape[0] != D or inv_cov_arr.shape[1] != D:
                raise ValueError(f"inv_cov must be ({D},{D}) for Mahalanobis.")
        elif metric == "minkowski":
            if inv_cov_arr.shape[0] != 1 or inv_cov_arr.shape[1] != 1:
                raise ValueError("For Minkowski distance, inv_cov must be a 1x1 array with param[0,0] = k.")
    
    # Allocate output arrays
    cdef np.ndarray[np.int32_t, ndim=2] indices = np.zeros((N, k), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((N, k), dtype=np.float64)
    
    # Compute k-nearest neighbors for each point - using regular for loop instead of prange
    for i in range(N):
        compute_knn_for_point(X, i, effective_k, indices, distances, metric, inv_cov_arr, include_self)
    
    return indices, distances

# Function to compute k-nearest neighbors for a single point
cpdef compute_knn_for_point(np.ndarray[np.float64_t, ndim=2] X, int i, int k,
                         np.ndarray[np.int32_t, ndim=2] indices,
                         np.ndarray[np.float64_t, ndim=2] distances,
                         str metric, np.ndarray[np.float64_t, ndim=2] inv_cov_arr,
                         bint include_self):
    """
    Compute k-nearest neighbors for a single point.
    
    Parameters:
    - X: Data matrix (N x D)
    - i: Index of the query point
    - k: Number of neighbors to find
    - indices: Output array for neighbor indices
    - distances: Output array for neighbor distances
    - metric: Distance metric to use
    - inv_cov_arr: Parameter for certain distance metrics
    - include_self: Whether to include self as a neighbor
    """
    cdef int N = X.shape[0]
    cdef int j, l, m
    cdef double dist
    cdef neighbor_t* neighbors = <neighbor_t*>malloc(N * sizeof(neighbor_t))
    cdef dist_func_t dist_func = _get_distance_function(metric)
    cdef double[:, ::1] inv_cov_view = inv_cov_arr
    
    # Compute distances to all other points
    for j in range(N):
        dist = dist_func(X[i], X[j], inv_cov_view)
        neighbors[j].idx = j
        neighbors[j].dist = dist
    
    # Sort neighbors by distance (simple insertion sort)
    cdef neighbor_t temp
    for j in range(1, N):
        temp = neighbors[j]
        l = j - 1
        while l >= 0 and neighbors[l].dist > temp.dist:
            neighbors[l + 1] = neighbors[l]
            l -= 1
        neighbors[l + 1] = temp
    
    # Copy k nearest neighbors to output arrays
    cdef int offset = 0 if include_self else 1
    cdef int actual_k = min(k, N - offset)
    
    for j in range(actual_k):
        indices[i, j] = neighbors[j + offset].idx
        distances[i, j] = neighbors[j + offset].dist
    
    # Fill remaining slots if needed
    for j in range(actual_k, k):
        indices[i, j] = -1
        distances[i, j] = -1.0
    
    free(neighbors)

cpdef object build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int k,
    object inv_cov=None,
    str metric="euclidean",
    str method="brute_force",
    str graph_type="knn"
):
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
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t i, j, l
    cdef np.ndarray[np.int32_t, ndim=2] knn_indices
    cdef np.ndarray[np.float64_t, ndim=2] distances
    cdef list edges
    cdef dist_func_t dist_func
    cdef bint is_gabriel
    cdef double d_ik, d_jk
    cdef double[:, ::1] X_view = X
    cdef double[:, ::1] inv_cov_view
    
    # Validate k
    if k >= N:
        raise ValueError(f"k ({k}) must be less than the number of data points ({N})")
    
    distances = np.full((N, N), np.inf, dtype=np.float64)
    edges = []  # List of (i, j, distance) tuples for sparse storage
    
    # Get the appropriate distance function and process inv_cov
    # Use the internal _get_distance_function directly for Cython code
    dist_func = _get_distance_function(metric)
    
    # Process inv_cov
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    if inv_cov is None:
        if metric == "mahalanobis":
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        elif metric == "minkowski":
            # Default to Euclidean distance (p=2) if not specified
            inv_cov_arr = np.zeros((1, 1), dtype=np.float64)
            inv_cov_arr[0, 0] = 2.0
        else:
            inv_cov_arr = np.zeros((1, 1), dtype=np.float64)
    else:
        inv_cov_arr = inv_cov
        
        # Validate inv_cov dimensions
        if metric == "mahalanobis":
            if inv_cov_arr.shape[0] != D or inv_cov_arr.shape[1] != D:
                raise ValueError(f"inv_cov must be ({D},{D}) for Mahalanobis.")
        elif metric == "minkowski":
            if inv_cov_arr.shape[0] != 1 or inv_cov_arr.shape[1] != 1:
                raise ValueError("For Minkowski distance, inv_cov must be a 1x1 array with param[0,0] = k.")
    
    inv_cov_view = inv_cov_arr

    if method == "kd_tree" and metric in ["euclidean", "manhattan"]:
        tree = cKDTree(X)
        for i in range(N):
            dists, idxs = tree.query(X[i], k+1)  # k+1 to include self
            # Skip self (first element, which has distance 0)
            for j in range(1, k+1):
                edges.append((i, idxs[j], dists[j]))
        
    else:
        # Compute pairwise distances (without parallelism for now)
        for i in range(N):
            for j in range(i + 1, N):
                distances[i, j] = dist_func(X_view[i], X_view[j], inv_cov_view)
                distances[j, i] = distances[i, j]  # Symmetric matrix
        
        if graph_type == "knn":
            # Find k nearest neighbors for each point using argpartition (O(N) instead of O(N log N))
            knn_indices = np.zeros((N, k), dtype=np.int32)
            for i in range(N):
                sorted_indices = np.argpartition(distances[i], k + 1)[:k + 1]
                knn_indices[i, :] = sorted_indices[1 : k + 1]  # Exclude self
            
            # Store k-NN edges
            for i in range(N):
                for j in range(k):
                    neighbor_idx = knn_indices[i, j]
                    edges.append((i, neighbor_idx, distances[i, neighbor_idx]))
        
        elif graph_type == "gabriel":
            # Construct Gabriel Graph
            for i in range(N):
                for j in range(i + 1, N):
                    is_gabriel = True
                    for l in range(N):
                        if l != i and l != j:
                            d_ik = distances[i, l]
                            d_jk = distances[j, l]
                            if d_ik < distances[i, j] and d_jk < distances[i, j]:
                                is_gabriel = False
                                break
                    if is_gabriel:
                        edges.append((i, j, distances[i, j]))
                        edges.append((j, i, distances[i, j]))
    
    # Convert to sparse CSR matrix
    row_indices, col_indices, values = zip(*edges) if edges else ([], [], [])
    adjacency_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
    
    return adjacency_matrix
