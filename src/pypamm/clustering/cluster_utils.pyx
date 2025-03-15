# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp
from cython.parallel import prange
from pypamm.distance_metrics cimport dist_mahalanobis

# ------------------------------------------------------------------------------
# 1. Cluster Covariance Computation
# ------------------------------------------------------------------------------
cpdef np.ndarray[np.float64_t, ndim=3] compute_cluster_covariance(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels
):
    """
    Computes covariance matrices for each cluster.

    Parameters:
    - X: (N x D) NumPy array of data points.
    - cluster_labels: (N,) array of cluster assignments.

    Returns:
    - cov_matrices: (num_clusters x D x D) covariance matrices.
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t num_clusters = np.max(cluster_labels) + 1
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t cluster_id

    # Initialize covariance matrices
    cdef np.ndarray[np.float64_t, ndim=3] cov_matrices = np.zeros((num_clusters, D, D), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] cluster_sizes = np.zeros(num_clusters, dtype=np.int32)

    # Compute mean vectors
    cdef np.ndarray[np.float64_t, ndim=2] cluster_means = np.zeros((num_clusters, D), dtype=np.float64)
    
    for i in range(N):
        cluster_id = cluster_labels[i]
        cluster_means[cluster_id] += X[i]
        cluster_sizes[cluster_id] += 1
    
    for i in range(num_clusters):
        if cluster_sizes[i] > 0:
            cluster_means[i] /= cluster_sizes[i]  # Normalize means

    # Compute diagonal covariance matrices (variances along each dimension)
    for i in range(N):
        cluster_id = cluster_labels[i]
        for j in range(D):
            # Only compute diagonal elements (variances)
            cov_matrices[cluster_id, j, j] += (X[i, j] - cluster_means[cluster_id, j]) ** 2
    
    # Normalize
    for i in range(num_clusters):
        if cluster_sizes[i] > 1:
            # Apply Bessel's correction
            for j in range(D):
                cov_matrices[i, j, j] /= (cluster_sizes[i] - 1)

    return cov_matrices

# ------------------------------------------------------------------------------
# 2. Log-Sum-Exp for Cluster Merging
# ------------------------------------------------------------------------------
cdef double logsumexp(np.ndarray[np.float64_t, ndim=1] arr):
    """
    Computes log-sum-exp in a numerically stable way.
    Used for probability-based cluster merging.
    """
    cdef double max_val = np.max(arr)
    return max_val + log(np.sum(np.exp(arr - max_val)))

# ------------------------------------------------------------------------------
# 3. Cluster Merging
# ------------------------------------------------------------------------------
cpdef np.ndarray[np.int32_t, ndim=1] merge_clusters(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=1] prob,
    np.ndarray[np.int32_t, ndim=1] cluster_labels,
    np.ndarray[np.float64_t, ndim=3] cluster_covariances,
    double threshold=0.8
):
    """
    Merges weak clusters based on probability and adjacency.

    Parameters:
    - X: (N x D) Data points.
    - prob: (N,) Probabilities associated with each data point.
    - cluster_labels: (N,) Cluster assignments.
    - cluster_covariances: (num_clusters x D x D) Covariance matrices per cluster.
    - threshold: Probability threshold below which a cluster should be merged.

    Returns:
    - Updated cluster_labels.
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t num_clusters = np.max(cluster_labels) + 1
    cdef Py_ssize_t i, c1, c2
    cdef np.ndarray[np.float64_t, ndim=1] cluster_probs
    cdef np.ndarray[np.uint8_t, ndim=1] mergeornot
    cdef np.ndarray[np.int32_t, ndim=1] new_labels
    cdef double normpks

    # Track which clusters need merging
    mergeornot = np.zeros(num_clusters, dtype=np.uint8)
    normpks = logsumexp(prob)

    # Calculate cluster probabilities
    cluster_probs = np.zeros(num_clusters, dtype=np.float64)
    for c1 in range(num_clusters):
        mask = cluster_labels == c1
        if np.any(mask):
            cluster_probs[c1] = np.sum(prob[mask]) / np.sum(prob)
    
    # Identify weak clusters to merge
    for c1 in range(num_clusters):
        if cluster_probs[c1] < threshold:
            mergeornot[c1] = 1  # Mark for merging

    # Create a copy of the labels to avoid modifying during iteration
    new_labels = cluster_labels.copy()
    
    # Merge weak clusters
    for i in range(N):
        c1 = cluster_labels[i]
        if mergeornot[c1]:
            new_labels[i] = _find_nearest_cluster(i, X, cluster_labels, cluster_covariances, mergeornot)

    return new_labels

# ------------------------------------------------------------------------------
# 4. Helper Function: Find Nearest Cluster
# ------------------------------------------------------------------------------
cdef int _find_nearest_cluster(
    int i,
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels,
    np.ndarray[np.float64_t, ndim=3] cluster_covariances,
    np.ndarray[np.uint8_t, ndim=1] mergeornot
):
    """
    Finds the nearest valid cluster to merge into using Mahalanobis distance.

    Parameters:
    - i: Index of the point to reassign.
    - X: (N x D) Data points.
    - cluster_labels: (N,) Cluster assignments.
    - cluster_covariances: (num_clusters x D x D) Covariance matrices.
    - mergeornot: (num_clusters,) Boolean array indicating which clusters should be merged.

    Returns:
    - Nearest cluster ID.
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t num_clusters = np.max(cluster_labels) + 1
    cdef Py_ssize_t j, c1, c2
    cdef double min_dist = np.inf
    cdef int best_cluster = cluster_labels[i]
    cdef double d
    cdef np.ndarray cluster_indices_np
    cdef np.ndarray[np.float64_t, ndim=1] center
    cdef double[::1] x_i = X[i]
    
    # Skip if the cluster doesn't need merging
    c1 = cluster_labels[i]
    if not mergeornot[c1]:
        return c1
    
    # Calculate distance to each cluster center
    for c2 in range(num_clusters):
        # Skip the current cluster and other clusters that need merging
        if c2 == c1 or mergeornot[c2]:
            continue
            
        # Get points in this cluster
        cluster_indices_np = np.where(cluster_labels == c2)[0]
        if len(cluster_indices_np) == 0:
            continue
            
        # Calculate cluster center
        center = np.mean(X[cluster_indices_np], axis=0)
        
        # Calculate distance to center using the cluster's covariance
        d = dist_mahalanobis(x_i, center, cluster_covariances[c2])
        
        if d < min_dist:
            min_dist = d
            best_cluster = c2

    return best_cluster 