# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport log, exp

# Import distance functions
from pypamm.distance_metrics cimport dist_func_t, _get_distance_function


def quick_shift_clustering(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=1] prob,
    int ngrid,
    str metric="euclidean",
    double lambda_qs=1.0,
    double max_dist=np.inf
):
    """
    Quick-Shift clustering algorithm based on density gradient ascent.
    
    Parameters:
    - X: (N x D) NumPy array (data points)
    - prob: (N,) NumPy array of probability estimates for each point
    - ngrid: Number of grid points
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski")
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold for connecting points (default: infinity)

    Returns:
    - idxroot: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[np.int32_t, ndim=1] idxroot = np.arange(N, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] distmm = np.full((N, N), np.inf, dtype=np.float64)
    
    # Select the appropriate distance function
    cdef dist_func_t dist_func = _get_distance_function(metric)
    
    # Compute pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            distmm[i, j] = dist_func(X[i], X[j], np.zeros((1, 1)))
            distmm[j, i] = distmm[i, j]
    
    # Quick-Shift algorithm: Find the nearest point with higher probability
    for i in range(N):
        min_dist = np.inf
        best_j = i  # Default: stay in own cluster
        for j in range(N):
            if prob[j] > prob[i]:  # Only move towards denser regions
                d = distmm[i, j]
                if d < min_dist and d <= max_dist:
                    min_dist = d
                    best_j = j
        idxroot[i] = best_j  # Assign nearest high-density neighbor
    
    # Propagate cluster assignments to find root nodes
    # This ensures that all points in a path connect to the same root
    cdef np.ndarray[np.int32_t, ndim=1] idxroot_final = np.zeros(N, dtype=np.int32)
    for i in range(N):
        j = i
        path = []
        # Follow the path until we reach a point that points to itself
        while idxroot[j] != j and len(path) < N:  # Prevent infinite loops
            path.append(j)
            j = idxroot[j]
        # Assign all points in the path to the same root
        for p in path:
            idxroot_final[p] = j
        # Also assign the root itself
        idxroot_final[j] = j
    
    # Extract cluster centers
    cluster_centers = np.unique(idxroot_final)
    
    return idxroot_final, cluster_centers