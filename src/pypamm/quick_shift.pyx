# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport log, exp, sqrt, fabs
from scipy.sparse import csr_matrix

# Import distance functions
from pypamm.distance_metrics cimport dist_func_t, _get_distance_function

def quick_shift_clustering(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=1] prob,
    int ngrid,
    object neighbor_graph=None,  # Graph-based constraints (MST, k-NN, Gabriel)
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
    - lambda_qs: Scaling factor for density-based traversal (default: 1.0)
    - max_dist: Maximum distance threshold (default: np.inf)

    Returns:
    - cluster_labels: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t i, j, d, cluster_id

    # Initialize arrays for tracking cluster assignments
    cdef np.ndarray[np.int32_t, ndim=1] parents = np.arange(N, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] min_dist = np.full(N, np.inf, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] nearest_higher_density = np.full(N, -1, dtype=np.int32)

    # Select the appropriate distance function
    cdef dist_func_t dist_func = _get_distance_function(metric)

    # Create a dummy parameter for the distance function
    cdef np.ndarray[np.float64_t, ndim=2] dummy_param = np.zeros((1, 1), dtype=np.float64)
    cdef double dist_val

    # Create memoryviews for the distance function
    cdef double[::1] point_i, point_j
    cdef double[:, ::1] param_view = dummy_param

    # Sort points by density (higher density first)
    cdef np.ndarray[np.int64_t, ndim=1] sorted_indices = np.argsort(-prob)

    # Process points in order of decreasing density
    if neighbor_graph is not None:
        # Use neighbor graph for efficiency
        for idx in range(N):
            i = sorted_indices[idx]
            point_i = X[i]
            for j in neighbor_graph.indices[neighbor_graph.indptr[i]:neighbor_graph.indptr[i + 1]]:
                # Only consider neighbors with higher density (scaled by lambda_qs)
                if prob[j] > prob[i] * lambda_qs:
                    point_j = X[j]
                    dist_val = dist_func(point_i, point_j, param_view)
                    if dist_val < min_dist[i] and dist_val <= max_dist:
                        min_dist[i] = dist_val
                        nearest_higher_density[i] = j
    else:
        # Brute-force approach for all pairs
        for idx in range(N):
            i = sorted_indices[idx]
            point_i = X[i]
            for j in range(N):
                if i == j:
                    continue  # Skip self
                # Only consider points with higher density (scaled by lambda_qs)
                if prob[j] > prob[i] * lambda_qs:
                    point_j = X[j]
                    dist_val = dist_func(point_i, point_j, param_view)
                    if dist_val < min_dist[i] and dist_val <= max_dist:
                        min_dist[i] = dist_val
                        nearest_higher_density[i] = j

    # Assign parents based on nearest higher-density neighbor
    for i in range(N):
        if nearest_higher_density[i] != -1:
            parents[i] = nearest_higher_density[i]

    # Propagate cluster assignments to find roots
    cdef np.ndarray[np.int32_t, ndim=1] cluster_roots = np.zeros(N, dtype=np.int32)

    for i in range(N):
        if parents[i] == i:
            # This point is its own parent (a cluster root)
            cluster_roots[i] = i
        else:
            # Follow the path to the root
            current = i
            path = []

            # Traverse up to find the root (with cycle detection)
            while parents[current] != current and len(path) <= N:
                path.append(current)
                current = parents[current]

            # The root is found
            root = current

            # Path compression: update all nodes in the path to point directly to the root
            for node in path:
                parents[node] = root

            # Store the root
            cluster_roots[i] = root

    # Create final cluster labels
    unique_roots, inverse = np.unique(cluster_roots, return_inverse=True)
    cdef np.ndarray[np.int32_t, ndim=1] cluster_labels = inverse.astype(np.int32)

    # Calculate cluster centers (mean of points in each cluster)
    cdef int n_clusters = len(unique_roots)
    cdef np.ndarray[np.float64_t, ndim=2] cluster_centers = np.zeros((n_clusters, D), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] cluster_counts = np.zeros(n_clusters, dtype=np.int32)

    # Sum up points in each cluster
    for i in range(N):
        cluster_id = cluster_labels[i]
        for d in range(D):
            cluster_centers[cluster_id, d] += X[i, d]
        cluster_counts[cluster_id] += 1

    # Divide by count to get mean
    for i in range(n_clusters):
        if cluster_counts[i] > 0:
            for d in range(D):
                cluster_centers[i, d] /= cluster_counts[i]

    return cluster_labels, cluster_centers
