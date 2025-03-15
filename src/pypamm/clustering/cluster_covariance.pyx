# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange

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
        cluster_means[cluster_labels[i]] += X[i]
        cluster_sizes[cluster_labels[i]] += 1
    
    for i in range(num_clusters):
        if cluster_sizes[i] > 0:
            cluster_means[i] /= cluster_sizes[i]  # Normalize means

    # Compute covariance matrices
    for i in prange(N, nogil=True):  # Parallelized computation
        cluster_id = cluster_labels[i]
        for j in range(D):
            for k in range(D):
                cov_matrices[cluster_id, j, k] += (
                    (X[i, j] - cluster_means[cluster_id, j]) *
                    (X[i, k] - cluster_means[cluster_id, k])
                )

    # Normalize covariance matrices
    for i in range(num_clusters):
        if cluster_sizes[i] > 1:
            cov_matrices[i] /= (cluster_sizes[i] - 1)  # Bessel's correction

    return cov_matrices