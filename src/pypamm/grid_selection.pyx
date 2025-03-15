# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, pow

# We'll use a unified function pointer signature:
#     double f(a, b, mat)
#   where "mat" can be an inverse covariance or a 1×1 array for Minkowski exponent.

ctypedef double (*dist_func_t)(double[::1], double[::1], double[:, ::1]) nogil

# ------------------------------------------------------------------------------
# 1) Distances that ignore the third parameter
# ------------------------------------------------------------------------------
cdef inline double dist_euclidean(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) nogil:
    """
    Squared Euclidean distance: sum_i (a[i] - b[i])^2
    (If you want the actual L2 distance, you'd do sqrt at the end,
     but we've historically used squared in these examples.)
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, dist_sq = 0.0
    for i in range(D):
        diff = a[i] - b[i]
        dist_sq += diff * diff
    return dist_sq


cdef inline double dist_manhattan(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) nogil:
    """
    Manhattan (L1) distance = sum_i |a[i] - b[i]|.
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, dist_val = 0.0
    for i in range(D):
        diff = a[i] - b[i]
        dist_val += fabs(diff)
    return dist_val


cdef inline double dist_chebyshev(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) nogil:
    """
    Chebyshev (L∞) distance = max_i |a[i] - b[i]|.
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, max_diff = 0.0
    for i in range(D):
        diff = fabs(a[i] - b[i])
        if diff > max_diff:
            max_diff = diff
    return max_diff


cdef inline double dist_cosine(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) nogil:
    """
    Cosine distance = 1 - (a·b)/(||a|| * ||b||).
    If either vector has length zero, define distance = 1.
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
# 2) Distances that USE the third parameter
# ------------------------------------------------------------------------------
cdef inline double dist_mahalanobis(
    double[::1] a,
    double[::1] b,
    double[:, ::1] inv_cov
) nogil:
    """
    Squared Mahalanobis distance = (a - b)^T inv_cov (a - b).
    'inv_cov' must be a D x D inverse covariance matrix.
    """
    cdef Py_ssize_t i, j, D = a.shape[0]
    cdef double diff_i, diff_j, dsum = 0.0
    for i in range(D):
        diff_i = a[i] - b[i]
        for j in range(D):
            diff_j = a[j] - b[j]
            dsum += diff_i * inv_cov[i, j] * diff_j
    return dsum


cdef inline double dist_minkowski(
    double[::1] a,
    double[::1] b,
    double[:, ::1] param
) nogil:
    """
    Minkowski distance with exponent k = param[0, 0].
    L_k(a, b) = ( sum_i |a[i] - b[i]|^k )^(1/k)
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double k = param[0, 0]
    cdef double diff, accum = 0.0
    for i in range(D):
        diff = fabs(a[i] - b[i])
        accum += pow(diff, k)
    return pow(accum, 1.0 / k)


# ------------------------------------------------------------------------------
# 3) Min–Max selection with flexible metrics
# ------------------------------------------------------------------------------
cpdef object select_grid_points(
    object X,  # np.ndarray[np.float64_t, ndim=2]
    int ngrid,
    str metric = "euclidean",
    object inv_cov = None  # np.ndarray[np.float64_t, ndim=2]
):
    """
    Select 'ngrid' points from X (N x D) by the min–max algorithm, using one
    of multiple metrics:
      - 'euclidean': squared Euclidean distance
      - 'manhattan': L1 distance
      - 'chebyshev': L∞ distance
      - 'cosine':    1 - cos_sim
      - 'mahalanobis': needs inv_cov = (D x D)
      - 'minkowski': needs inv_cov = (1 x 1) with inv_cov[0,0] = exponent k

    Returns (idxgrid, Y) where:
      - idxgrid: indices of chosen points (shape = [ngrid])
      - Y: coordinates of chosen points (shape = [ngrid, D])
    """
    # Type the arrays inside the function
    cdef np.ndarray[np.float64_t, ndim=2] X_arr = X
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    
    cdef Py_ssize_t N = X_arr.shape[0]
    cdef Py_ssize_t D = X_arr.shape[1]
    
    cdef dist_func_t dist_func

    # Decide which distance function to use
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
        inv_cov_arr = inv_cov
        if inv_cov_arr.shape[0] != D or inv_cov_arr.shape[1] != D:
            raise ValueError(f"inv_cov must be ({D},{D}) for Mahalanobis.")
        dist_func = dist_mahalanobis
    elif metric == "minkowski":
        if inv_cov is None:
            raise ValueError("Must supply a 1x1 array with exponent for Minkowski.")
        inv_cov_arr = inv_cov
        if inv_cov_arr.shape[0] != 1 or inv_cov_arr.shape[1] != 1:
            raise ValueError("For Minkowski distance, inv_cov must be a 1x1 array with param[0,0] = k.")
        dist_func = dist_minkowski
    else:
        raise ValueError(f"Unsupported metric '{metric}'")

    # Allocate result arrays
    cdef np.ndarray[np.int32_t, ndim=1] idxgrid = np.empty(ngrid, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] Y = np.empty((ngrid, D), dtype=np.float64)
    
    # Track the min distance of each point to any chosen grid point
    cdef np.ndarray[np.float64_t, ndim=1] dmin_ = np.empty(N, dtype=np.float64)

    # 1. Pick the first grid point randomly
    cdef int irandom = int(np.random.rand() * N)
    idxgrid[0] = irandom
    Y[0, :] = X_arr[irandom, :]

    # Initialize dmin_ using the first chosen grid point
    cdef Py_ssize_t i, j
    cdef double max_min_dist, d
    
    # Create a dummy parameter for metrics that don't use inv_cov
    cdef np.ndarray[np.float64_t, ndim=2] dummy_param
    if inv_cov is None:
        dummy_param = np.empty((1, 1), dtype=np.float64)
        if metric == "minkowski":
            # Default to Euclidean distance (p=2) if not specified
            dummy_param[0, 0] = 2.0
    
    for j in range(N):
        if inv_cov is not None:
            dmin_[j] = dist_func(X_arr[j, :], Y[0, :], inv_cov_arr)
        else:
            dmin_[j] = dist_func(X_arr[j, :], Y[0, :], dummy_param)

    # 2. Iteratively pick next points
    cdef Py_ssize_t jmax
    for i in range(1, ngrid):
        max_min_dist = -1.0
        jmax = -1
        for j in range(N):
            if dmin_[j] > max_min_dist:
                max_min_dist = dmin_[j]
                jmax = j
        
        # jmax is our new grid point
        idxgrid[i] = jmax
        Y[i, :] = X_arr[jmax, :]
        
        # Update dmin_ with this newly selected point
        for j in range(N):
            if inv_cov is not None:
                d = dist_func(X_arr[j, :], Y[i, :], inv_cov_arr)
            else:
                d = dist_func(X_arr[j, :], Y[i, :], dummy_param)
            if d < dmin_[j]:
                dmin_[j] = d

    return idxgrid, Y