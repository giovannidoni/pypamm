# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

# Import distance functions from the distance_metrics module
from pypamm.distance_metrics cimport (
    dist_func_t,
    dist_euclidean,
    dist_manhattan,
    dist_chebyshev,
    dist_cosine,
    dist_mahalanobis,
    dist_minkowski,
    _get_distance_function
)
from pypamm.distance_metrics import get_distance_function

# ------------------------------------------------------------------------------
# Min-Max selection with flexible metrics
# ------------------------------------------------------------------------------
cpdef object select_grid_points(
    double[:, :] X,
    int ngrid,
    str metric = "euclidean",
    object inv_cov = None
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
    # Create a C-contiguous copy of X to ensure consistent memory access
    cdef np.ndarray[np.float64_t, ndim=2] X_arr = np.ascontiguousarray(X)

    cdef Py_ssize_t N = X_arr.shape[0]
    cdef Py_ssize_t D = X_arr.shape[1]

    # Get the appropriate distance function and process inv_cov
    cdef object dist_func_wrapper
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov_arr
    dist_func_wrapper, inv_cov_arr = get_distance_function(metric, inv_cov, D)
    cdef dist_func_t dist_func = _get_distance_function(metric)

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

    for j in range(N):
        dmin_[j] = dist_func(X_arr[j, :], Y[0, :], inv_cov_arr)

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
            d = dist_func(X_arr[j, :], Y[i, :], inv_cov_arr)
            if d < dmin_[j]:
                dmin_[j] = d

    return idxgrid, Y
