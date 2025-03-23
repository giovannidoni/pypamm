# distance_metrics.pyx
# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython implementation of distance metrics for efficient computation.
This module provides optimized distance functions for various metrics.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, pow, HUGE_VAL

# ------------------------------------------------------------------------------
# Distance functions that ignore the third parameter
# ------------------------------------------------------------------------------
cdef double dist_euclidean(
    double[:] a,
    double[:] b,
) except? -1 nogil:
    """
    Squared Euclidean distance calculation.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)

    Returns:
    - Squared Euclidean distance: sum_i (a[i] - b[i])^2

    Notes:
    - Returns squared distance (not actual L2 norm)
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, dist_sq = 0.0
    for i in range(D):
        diff = a[i] - b[i]
        dist_sq += diff * diff
    return dist_sq


cdef double dist_manhattan(
    double[:] a,
    double[:] b,
) except? -1 nogil:
    """
    Manhattan (L1) distance calculation.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)

    Returns:
    - Manhattan distance: sum_i |a[i] - b[i]|

    Notes:
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, dist_val = 0.0
    for i in range(D):
        diff = a[i] - b[i]
        dist_val += fabs(diff)
    return dist_val


cdef double dist_chebyshev(
    double[:] a,
    double[:] b,
) except? -1 nogil:
    """
    Chebyshev (L∞) distance calculation.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)

    Returns:
    - Chebyshev distance: max_i |a[i] - b[i]|

    Notes:
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, max_diff = 0.0
    for i in range(D):
        diff = fabs(a[i] - b[i])
        if diff > max_diff:
            max_diff = diff
    return max_diff


cdef double dist_cosine(
    double[:] a,
    double[:] b,
) except? -1 nogil:
    """
    Cosine distance calculation.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)

    Returns:
    - Cosine distance: 1 - (a·b)/(||a|| * ||b||)

    Notes:
    - If either vector has zero length, returns 1.0
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double dot_ab = 0.0
    cdef double norm_a = 0.0
    cdef double norm_b = 0.0
    for i in range(D):
        dot_ab += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cdef double denom = sqrt(norm_a) * sqrt(norm_b)
    cdef double cos_sim = dot_ab / denom
    return 1.0 - cos_sim


# ------------------------------------------------------------------------------
# Distance functions that USE the third parameter
# ------------------------------------------------------------------------------
cdef double dist_mahalanobis(
    double[:] a,
    double[:] b,
    double[:, :] inv_cov
) except? -1 nogil:
    """
    Squared Mahalanobis distance calculation.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)
    - inv_cov: Inverse covariance matrix (D x D)

    Returns:
    - Squared Mahalanobis distance: (a - b)^T inv_cov (a - b)

    Notes:
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, j, D = a.shape[0]
    cdef double diff_i, diff_j, dsum = 0.0
    for i in range(D):
        diff_i = a[i] - b[i]
        for j in range(D):
            diff_j = a[j] - b[j]
            dsum += diff_i * inv_cov[i, j] * diff_j
    return dsum


cdef double dist_minkowski(
    double[:] a,
    double[:] b,
    int k
) except? -1 nogil:
    """
    Minkowski distance calculation with exponent k.

    Parameters:
    - a: First vector (D,)
    - b: Second vector (D,)
    - k: Power parameter (p-norm)

    Returns:
    - Minkowski distance: (sum_i |a[i] - b[i]|^k)^(1/k)

    Notes:
    - When k=1, this is equivalent to Manhattan distance
    - When k=2, this is equivalent to Euclidean distance
    - When k=∞, this is equivalent to Chebyshev distance
    - Returns -1 only in case of error
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, accum = 0.0
    for i in range(D):
        diff = fabs(a[i] - b[i])
        accum += pow(diff, k)
    return pow(accum, 1.0 / k)

# Internal function to calculate distance given the metric
cdef double calculate_distance(str metric, double[:] a, double[:] b, int k = 2, object inv_cov = None) except *:
    """
    Calculate distance between two vectors using the specified metric.

    Parameters:
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev",
              "cosine", "mahalanobis", "minkowski")
    - a: First vector
    - b: Second vector
    - k: Parameter for Minkowski distance (p value), default is 2
    - inv_cov: Optional inverse covariance matrix for mahalanobis distance

    Returns:
    - Distance value according to the specified metric

    Raises:
    - ValueError: If an unsupported metric is specified or required parameters are missing
    """
    # For metrics that don't require additional parameters
    cdef double[:, :] inv_cov_view

    if metric == "euclidean":
        return dist_euclidean(a, b)
    elif metric == "manhattan":
        return dist_manhattan(a, b)
    elif metric == "chebyshev":
        return dist_chebyshev(a, b)
    elif metric == "cosine":
        return dist_cosine(a, b)
    # For metrics that require additional parameters
    elif metric == "mahalanobis":
        if inv_cov is None:
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        # We need to convert inv_cov to the proper type
        inv_cov_view = inv_cov
        return dist_mahalanobis(a, b, inv_cov_view)
    elif metric == "minkowski":
        return dist_minkowski(a, b, k)
    else:
        raise ValueError(f"Unsupported metric '{metric}'")

cpdef double py_calculate_distance(str metric, double[:] a, double[:] b, int k = 2, object inv_cov = None) except *:
    """
    Python-accessible function to calculate distance between two vectors.

    Parameters:
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev",
              "cosine", "mahalanobis", "minkowski")
    - a: First vector
    - b: Second vector
    - k: Parameter for Minkowski distance (p value), default is 2
    - inv_cov: Optional inverse covariance matrix for mahalanobis distance

    Returns:
    - Distance value according to the specified metric

    Raises:
    - ValueError: If an unsupported metric is specified or required parameters are missing
    """
    # For metrics that don't require additional parameters
    return calculate_distance(metric, a, b, k, inv_cov)


cpdef tuple compute_pairwise_distances(
    double[:, :] a,
    str metric = "euclidean",
    int k = 2,
    object inv_cov = None,
):
    """
    Compute pairwise distances between all points in the input array.

    Parameters:
    - a: Data matrix (ngrid x D) of points
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev",
              "cosine", "mahalanobis", "minkowski")
    - k: Parameter for Minkowski distance (p value), default is 2
    - inv_cov: Optional inverse covariance matrix for mahalanobis distance

    Returns:
    - dist_mat: Distance matrix (ngrid x ngrid) between all pairs of points
    - min_dist: Minimum distance (ngrid,) from each point to any other point
    - min_dist_id: Index (ngrid,) of the closest point for each point
    """
    cdef int D = a.shape[1]
    cdef int ngrid = a.shape[0]
    cdef int i, j
    cdef double d

    cdef np.ndarray[np.float64_t, ndim=2] dist_mat = np.empty((ngrid, ngrid), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] min_dist = np.zeros(ngrid, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] min_dist_id = np.zeros(ngrid, dtype=np.int32)

    # Initialize minimum distances
    for i in range(ngrid):
        min_dist[i] = HUGE_VAL

    for i in range(ngrid):
        for j in range(i):
            d = calculate_distance(metric, a[i, :], a[j, :], k, inv_cov)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

            if d < min_dist[i]:
                min_dist[i] = d
                min_dist_id[i] = j

            if d < min_dist[j]:
                min_dist[j] = d
                min_dist_id[j] = i

    return dist_mat, min_dist, min_dist_id
