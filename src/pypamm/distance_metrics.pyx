# distance_metrics.pyx
# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, pow

# Import the function pointer type from the .pxd file
from pypamm.distance_metrics cimport dist_func_t

# ------------------------------------------------------------------------------
# 1) Distances that ignore the third parameter
# ------------------------------------------------------------------------------
cdef double dist_euclidean(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil:
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


cdef double dist_manhattan(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil:
    """
    Manhattan (L1) distance = sum_i |a[i] - b[i]|.
    """
    cdef Py_ssize_t i, D = a.shape[0]
    cdef double diff, dist_val = 0.0
    for i in range(D):
        diff = a[i] - b[i]
        dist_val += fabs(diff)
    return dist_val


cdef double dist_chebyshev(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil:
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


cdef double dist_cosine(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil:
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
cdef double dist_mahalanobis(
    double[::1] a,
    double[::1] b,
    double[:, ::1] inv_cov
) except? -1 nogil:
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


cdef double dist_minkowski(
    double[::1] a,
    double[::1] b,
    double[:, ::1] param
) except? -1 nogil:
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

# Function to select the appropriate distance function
def get_distance_function(str metric, object inv_cov=None, int D=0):
    """
    Select the appropriate distance function based on the metric.
    
    Parameters:
    - metric: String specifying the distance metric
    - inv_cov: Optional parameter for Mahalanobis and Minkowski distances
    - D: Dimensionality of the data (needed for validation)
    
    Returns:
    - dist_func: A Python wrapper function for the selected distance function
    - inv_cov_arr: The processed inv_cov parameter (or a dummy if not needed)
    """
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    
    # Select appropriate distance function
    if metric == "euclidean":
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_euclidean(a, b, inv_cov_param)
    elif metric == "manhattan":
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_manhattan(a, b, inv_cov_param)
    elif metric == "chebyshev":
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_chebyshev(a, b, inv_cov_param)
    elif metric == "cosine":
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_cosine(a, b, inv_cov_param)
    elif metric == "mahalanobis":
        if inv_cov is None:
            raise ValueError("Must supply inv_cov (D x D) for Mahalanobis.")
        inv_cov_arr = inv_cov
        if D > 0 and (inv_cov_arr.shape[0] != D or inv_cov_arr.shape[1] != D):
            raise ValueError(f"inv_cov must be ({D},{D}) for Mahalanobis.")
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_mahalanobis(a, b, inv_cov_param)
    elif metric == "minkowski":
        if inv_cov is None:
            raise ValueError("Must supply a 1x1 array with exponent for Minkowski.")
        inv_cov_arr = inv_cov
        if inv_cov_arr.shape[0] != 1 or inv_cov_arr.shape[1] != 1:
            raise ValueError("For Minkowski distance, inv_cov must be a 1x1 array with param[0,0] = k.")
        def dist_func_wrapper(a, b, inv_cov_param):
            return dist_minkowski(a, b, inv_cov_param)
    else:
        raise ValueError(f"Unsupported metric '{metric}'")
    
    # Create a dummy parameter for metrics that don't use inv_cov
    if inv_cov is None:
        inv_cov_arr = np.zeros((1, 1), dtype=np.float64)
        if metric == "minkowski":
            # Default to Euclidean distance (p=2) if not specified
            inv_cov_arr[0, 0] = 2.0
    
    return dist_func_wrapper, inv_cov_arr

# Internal function to get the actual distance function (for Cython use only)
cdef dist_func_t _get_distance_function(str metric) except *:
    if metric == "euclidean":
        return dist_euclidean
    elif metric == "manhattan":
        return dist_manhattan
    elif metric == "chebyshev":
        return dist_chebyshev
    elif metric == "cosine":
        return dist_cosine
    elif metric == "mahalanobis":
        return dist_mahalanobis
    elif metric == "minkowski":
        return dist_minkowski
    else:
        raise ValueError(f"Unsupported metric '{metric}'") 