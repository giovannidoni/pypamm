# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport log, exp
from scipy.sparse import csr_matrix

# Import distance functions
from pypamm.distance_metrics cimport dist_func_t, _get_distance_function

def quick_shift_clustering(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=1] prob,
    int ngrid,
    object neighbor_graph=None,  # ✅ NEW: Graph-based constraints (MST, k-NN, Gabriel)
    str metric="euclidean",
    double lambda_qs=1.0,
    double max_dist=np.inf
):
    """
    Quick-Shift clustering algorithm with optional graph constraints.

    Parameters:
    - X: (N x D) NumPy array (data points)
    - prob: (N,) NumPy array of probability estimates for each point
    - ngrid: Number of grid points
    - neighbor_graph: Optional sparse graph (MST, k-NN, Gabriel Graph)
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", etc.)
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold

    Returns:
    - cluster_labels: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t i, j, d

    # Initialize cluster labels (each point starts as its own cluster)
    cdef np.ndarray[np.int32_t, ndim=1] idxroot = np.arange(N, dtype=np.int32)

    # Select the appropriate distance function
    cdef dist_func_t dist_func = _get_distance_function(metric)

    # Store minimum distance to nearest higher-density neighbor
    cdef np.ndarray[np.float64_t, ndim=1] min_dist = np.full(N, np.inf, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] nearest_neighbor = -1 * np.ones(N, dtype=np.int32)

    # Sort points by density (higher density first)
    sorted_indices = np.argsort(-prob)

    # Create a dummy parameter for the distance function
    cdef np.ndarray[np.float64_t, ndim=2] dummy_param = np.zeros((1, 1), dtype=np.float64)
    cdef double dist_val

    # Use neighbor graph if available
    if neighbor_graph is not None:
        for i in sorted_indices:
            for j in neighbor_graph.indices[neighbor_graph.indptr[i]:neighbor_graph.indptr[i + 1]]:
                if prob[j] > prob[i] * lambda_qs:  # Apply lambda_qs scaling factor
                    dist_val = dist_func(X[i], X[j], dummy_param)
                    if dist_val < min_dist[i] and dist_val <= max_dist:
                        min_dist[i] = dist_val
                        nearest_neighbor[i] = j
    else:
        # Brute-force method (if no graph provided)
        for i in sorted_indices:
            nearest_idx = -1
            min_dist_i = np.inf
            for j in range(N):
                if i == j:
                    continue  # Skip self
                if prob[j] > prob[i] * lambda_qs:  # Apply lambda_qs scaling factor
                    dist_val = dist_func(X[i], X[j], dummy_param)
                    if dist_val < min_dist_i and dist_val <= max_dist:
                        min_dist_i = dist_val
                        nearest_idx = j

            if nearest_idx != -1:
                min_dist[i] = min_dist_i
                nearest_neighbor[i] = nearest_idx

    # Assign cluster labels based on nearest higher-density neighbor
    for i in range(N):
        if nearest_neighbor[i] != -1:
            idxroot[i] = idxroot[nearest_neighbor[i]]

    # Get unique cluster roots
    unique_roots, inverse = np.unique(idxroot, return_inverse=True)

    # Create cluster labels from inverse mapping
    cdef np.ndarray[np.int32_t, ndim=1] cluster_labels = inverse.astype(np.int32)

    # Calculate cluster centers (mean of points in each cluster)
    cdef np.ndarray[np.float64_t, ndim=2] cluster_centers = np.zeros((len(unique_roots), D), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] cluster_counts = np.zeros(len(unique_roots), dtype=np.int32)

    # Sum up points in each cluster
    for i in range(N):
        cluster_id = cluster_labels[i]
        for d in range(D):
            cluster_centers[cluster_id, d] += X[i, d]
        cluster_counts[cluster_id] += 1

    # Divide by count to get mean
    for i in range(len(unique_roots)):
        if cluster_counts[i] > 0:
            for d in range(D):
                cluster_centers[i, d] /= cluster_counts[i]

    return cluster_labels, cluster_centers
