# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython implementation of neighbor graph construction
This module supports building k-nearest neighbor (KNN) and Gabriel graphs
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, pow
from scipy.spatial import cKDTree  # Faster neighbor search
from scipy.sparse import csr_matrix  # Sparse storage for adjacency graph
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# Import distance functions from the distance_metrics module
from pypamm.lib.distance cimport (
    calculate_distance
)

# Define a structure to hold a neighbor and its distance
ctypedef struct neighbor_t:
    int idx
    double dist

# Public API - this doesn't actually control what's exported in Cython
# The functions need to be properly defined and visible at the module level
__all__ = ['build_neighbor_graph', 'build_knn_graph', 'compute_knn_for_point']

# Function to build a k-nearest neighbor graph
cpdef tuple build_knn_graph(np.ndarray[np.float64_t, ndim=2] X, int n_neigh, str metric="euclidean", int k = 2,
                     object inv_cov=None, bint include_self=False, int n_jobs=-1):
    """
    Build a k-nearest neighbors (KNN) graph.

    Parameters:
    - X: Data matrix (N x D)
    - n_neigh: Number of neighbors
    - metric: Distance metric for neighbor calculation
    - k:  Exponent for the distance metric
    - inv_cov: Optional inverse covariance matrix for Mahalanobis distance
    - include_self: Whether to include self loops
    - n_jobs: Number of parallel jobs

    Returns:
    - indices: Indices of neighbors for each point (N x n_neigh)
    - distances: Distances to neighbors for each point (N x n_neigh)
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t i, j
    cdef np.ndarray[np.int32_t, ndim=2] indices = np.zeros((n, n_neigh), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((n, n_neigh), dtype=np.float64)

    # Prepare inverse covariance matrix view for distance calculation
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    if inv_cov is not None:
        inv_cov_arr = np.asarray(inv_cov, dtype=np.float64)
    else:
        inv_cov_arr = np.zeros((1, 1), dtype=np.float64)

    # Single-threaded implementation
    if n_jobs == 1:
        for i in range(n):
            compute_knn_for_point(X, i, n_neigh, indices, distances, metric, k, inv_cov_arr, include_self)
    else:
        # TODO: Implement multi-threaded version if needed
        for i in range(n):
            compute_knn_for_point(X, i, n_neigh, indices, distances, metric, k, inv_cov_arr, include_self)

    return indices, distances

# Function to compute k-nearest neighbors for a single point
cpdef compute_knn_for_point(np.ndarray[np.float64_t, ndim=2] X,
                         int i,
                         int n_neigh,
                         np.ndarray[np.int32_t, ndim=2] indices,
                         np.ndarray[np.float64_t, ndim=2] distances,
                         str metric,
                         int k,
                         np.ndarray[np.float64_t, ndim=2] inv_cov_arr,
                         bint include_self):
    """
    Compute k-nearest neighbors for a single point.

    Parameters:
    - X: Data matrix (N x D)
    - i: Index of the current point
    - n_neigh: Number of neighbors to find
    - indices: Output array for indices (N x n_neigh)
    - distances: Output array for distances (N x n_neigh)
    - metric: Distance metric
    - k: Exponent for the distance metric
    - inv_cov_arr: Inverse covariance matrix (for Mahalanobis)
    - include_self: Whether to include self loops
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t j, l
    cdef double dist
    cdef double[:] dists = np.full(n, np.inf, dtype=np.float64)
    cdef int[:] inds = np.arange(n, dtype=np.int32)
    cdef double[:, :] inv_cov_view = inv_cov_arr

    # Compute distances to all other points
    for j in range(n):
        if not include_self and i == j:
            dists[j] = np.inf
            continue

        dist = calculate_distance(metric, X[i], X[j], k, inv_cov_arr)
        dists[j] = dist

    # Sort distances and get k-nearest
    cdef double temp_dist
    cdef int temp_ind
    for j in range(n):
        for l in range(j + 1, n):
            if dists[j] > dists[l]:
                # Swap distances
                temp_dist = dists[j]
                dists[j] = dists[l]
                dists[l] = temp_dist

                # Swap indices
                temp_ind = inds[j]
                inds[j] = inds[l]
                inds[l] = temp_ind

    # Copy k nearest to output arrays
    for j in range(n_neigh):
        indices[i, j] = inds[j]
        distances[i, j] = dists[j]

cpdef object build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int n_neigh,
    str method="knn",
    str graph_type="connectivity",
    str metric="euclidean",
    int k = 2,
    object inv_cov=None,
):
    """
    Build a neighbor graph using the specified method.

    Parameters:
    - X: Data matrix (N x D)
    - n_neigh: Number of neighbors (for KNN)
    - method: Method to build the graph (knn, gabriel)
    - graph_type: Type of graph to return (connectivity, distance)
    - metric: Distance metric to use (euclidean, manhattan, etc.)
    - k: Exponent for the distance metric
    - inv_cov: Optional inverse covariance matrix for Mahalanobis distance


    Returns:
    - A sparse matrix representation of the graph (CSR format)
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t i, j, l
    cdef double[:, :] X_view = X
    cdef double[:, :] inv_cov_view
    cdef np.ndarray[np.int32_t, ndim=2] indices
    cdef np.ndarray[np.float64_t, ndim=2] nn_distances
    cdef list row_indices = []
    cdef list col_indices = []
    cdef list data = []
    cdef bint is_gabriel
    cdef double d_ij, d_ik, d_jk

    if inv_cov is not None:
        inv_cov_view = np.asarray(inv_cov, dtype=np.float64)
    else:
        inv_cov_view = np.zeros((1, 1), dtype=np.float64)

    # Initialize the adjacency matrix
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((n, n), dtype=np.float64)

    # Compute pairwise distances for the requested metric
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distances[i, j] = 0.0
            else:
                distances[i, j] = calculate_distance(metric, X_view[i], X_view[j], k=k, inv_cov=inv_cov_view)
                distances[j, i] = distances[i, j]  # Symmetric

    # Build the appropriate graph structure
    if method == "knn":
        # K-nearest neighbors graph
        indices, nn_distances = build_knn_graph(X, n_neigh, metric, k, inv_cov, False, 1)

        # Create the adjacency matrix from KNN indices
        for i in range(n):
            for j in range(n_neigh):
                row_indices.append(i)
                col_indices.append(indices[i, j])

                if graph_type == "distance":
                    data.append(nn_distances[i, j])
                else:  # "connectivity"
                    data.append(1.0)

        return csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    elif method == "gabriel":
        # Gabriel graph construction
        for i in range(n):
            for j in range(i+1, n):
                d_ij = distances[i, j]
                is_gabriel = True

                # Check Gabriel condition
                for l in range(n):
                    if l != i and l != j:
                        d_ik = distances[i, l]
                        d_jk = distances[j, l]

                        # If any point is inside the sphere with diameter (i,j)
                        if d_ik**2 + d_jk**2 <= d_ij**2:
                            is_gabriel = False
                            break

                if is_gabriel:
                    row_indices.append(i)
                    col_indices.append(j)
                    row_indices.append(j)
                    col_indices.append(i)

                    if graph_type == "distance":
                        data.append(d_ij)
                        data.append(d_ij)
                    else:  # "connectivity"
                        data.append(1.0)
                        data.append(1.0)

        return csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    else:
        raise ValueError(f"Unknown graph construction method: {method}")
