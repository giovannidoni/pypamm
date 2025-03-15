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
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", etc.)
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold

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

    # Sort points by density (higher density first)
    sorted_indices = np.argsort(-prob)

    # Quick-Shift clustering logic using provided density values
    for i in sorted_indices:
        nearest_idx = -1
        min_dist = np.inf

        for j in range(N):
            if i == j:
                continue  # Skip self
            if distmm[i, j] > max_dist:
                continue  # Skip distant points

            if prob[j] > prob[i]:  # Follow the density gradient
                if distmm[i, j] < min_dist:
                    min_dist = distmm[i, j]
                    nearest_idx = j

        if nearest_idx != -1:
            idxroot[i] = idxroot[nearest_idx]  # Assign cluster label

    return idxroot, np.unique(idxroot)