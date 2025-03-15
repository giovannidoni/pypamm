# neighbor_graph.pyx
# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from scipy.spatial import cKDTree  # Faster neighbor search

# Import distance functions from your Cython module
from pypamm.grid_selection cimport (
    dist_func_t,
    dist_euclidean,
    dist_manhattan,
    dist_chebyshev,
    dist_cosine,
    dist_mahalanobis,
    dist_minkowski
)


def build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int k,
    np.ndarray[np.float64_t, ndim=2] inv_cov = None,
    str metric="euclidean",
    str method="brute_force"
):
    """
    Build a k-NN Neighbor Graph using a specified distance metric.
    
    Parameters:
    - X: (N x D) NumPy array (data points)
    - k: Number of nearest neighbors to keep
    - inv_cov: (D x D) inverse covariance matrix for Mahalanobis distance (only needed for Mahalanobis and Minkowski)
    - metric: "euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski"
    - method: "brute_force" (default) or "kd_tree" for faster search

    Returns:
    - adjacency_list: list of lists, where adjacency_list[i] contains (neighbor_idx, distance)
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t i, j
    
    # Validate k
    if k >= N:
        raise ValueError(f"k ({k}) must be less than the number of data points ({N})")
    
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.full((N, N), np.inf, dtype=np.float64)
    cdef list adjacency_list = [[] for _ in range(N)]

    # Select appropriate distance function
    cdef dist_func_t dist_func
    if metric == "euclidean":
        dist_func = dist_euclidean
    elif metric == "manhattan":
        dist_func = dist_manhattan
    elif metric == "chebyshev":
        dist_func = dist_chebyshev
    elif metric == "cosine":
        dist_func = dist_cosine
    elif metric == "mahalanobis":
        if inv_cov is None:
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        dist_func = dist_mahalanobis
    elif metric == "minkowski":
        if inv_cov is None:
            raise ValueError("Must supply a 1x1 array with exponent for Minkowski.")
        dist_func = dist_minkowski
    else:
        raise ValueError(f"Unsupported metric '{metric}'")

    if method == "kd_tree" and metric in ["euclidean", "manhattan"]:
        tree = cKDTree(X)
        for i in range(N):
            dists, idxs = tree.query(X[i], k+1)  # k+1 to include self
            # Skip self (first element, which has distance 0)
            for j in range(1, k+1):
                adjacency_list[i].append((idxs[j], dists[j]))
        return adjacency_list

    # Compute pairwise distances (Brute Force method)
    for i in range(N):
        for j in range(N):  # Compute all pairwise distances
            if i != j:  # Skip self
                distances[i, j] = dist_func(X[i], X[j], inv_cov if inv_cov is not None else np.zeros((1,1)))
            else:
                distances[i, j] = np.inf  # Set self-distance to infinity to exclude self

    # Find k nearest neighbors for each point
    cdef np.ndarray[np.int32_t, ndim=2] knn_indices = np.zeros((N, k), dtype=np.int32)
    
    for i in range(N):
        sorted_indices = np.argsort(distances[i])  # Sort distances
        knn_indices[i, :] = sorted_indices[:k]  # Take k nearest (self is excluded due to inf distance)
    
    # Build adjacency list
    for i in range(N):
        for j in range(k):
            neighbor_idx = knn_indices[i, j]
            adjacency_list[i].append((neighbor_idx, distances[i, neighbor_idx]))
    
    return adjacency_list