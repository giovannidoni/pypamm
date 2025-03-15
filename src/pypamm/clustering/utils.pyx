# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport log, exp
from pypamm.distance_metrics cimport dist_mahalanobis

# ------------------------------------------------------------------------------
# 1. Log-Sum-Exp for Cluster Merging (Step 3)
# ------------------------------------------------------------------------------
cdef double logsumexp(np.ndarray[np.float64_t, ndim=1] arr):
    """
    Computes log-sum-exp in a numerically stable way.
    Used for probability-based cluster merging.
    """
    cdef double max_val = np.max(arr)
    return max_val + log(np.sum(np.exp(arr - max_val)))


# ------------------------------------------------------------------------------
# 2. Cluster Merging (Step 4)
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

    # Track which clusters need merging
    cdef np.ndarray[np.uint8_t, ndim=1] mergeornot = np.zeros(num_clusters, dtype=np.uint8)
    cdef double normpks = logsumexp(prob)

    # Identify weak clusters to merge
    for c1 in range(num_clusters):
        cluster_prob = logsumexp(prob[cluster_labels == c1])
        if exp(cluster_prob - normpks) < threshold:
            mergeornot[c1] = 1  # Mark for merging

    # Merge weak clusters
    for i in range(N):
        c1 = cluster_labels[i]
        if mergeornot[c1]:
            cluster_labels[i] = _find_nearest_cluster(i, X, cluster_labels, cluster_covariances)

    return cluster_labels


# ------------------------------------------------------------------------------
# 3. Helper Function: Find Nearest Cluster
# ------------------------------------------------------------------------------
cdef int _find_nearest_cluster(
    int i,
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels,
    np.ndarray[np.float64_t, ndim=3] cluster_covariances
):
    """
    Finds the nearest valid cluster to merge into using Mahalanobis distance.

    Parameters:
    - i: Index of the point to reassign.
    - X: (N x D) Data points.
    - cluster_labels: (N,) Cluster assignments.
    - cluster_covariances: (num_clusters x D x D) Covariance matrices.

    Returns:
    - Nearest cluster ID.
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t num_clusters = np.max(cluster_labels) + 1
    cdef Py_ssize_t j, c2
    cdef double min_dist = np.inf
    cdef int best_cluster = cluster_labels[i]
    cdef double d
    cdef double[::1] x_i = X[i]

    for j in range(N):
        if i == j:
            continue
        c2 = cluster_labels[j]
        if c2 == best_cluster:
            continue  # Skip the same cluster

        # Convert to memoryviews for the distance function
        d = dist_mahalanobis(x_i, X[j], cluster_covariances[c2])
        if d < min_dist:
            min_dist = d
            best_cluster = c2

    return best_cluster