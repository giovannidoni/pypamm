# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython implementation of grid selection algorithms.
This module provides efficient methods for selecting representative grid points from data.
"""

import numpy as np
cimport numpy as np

# Import distance functions from the lib module
from pypamm.lib.distance cimport (
    calculate_distance
)
from libc.math cimport HUGE_VAL

# ------------------------------------------------------------------------------
# Associate each point to the closet gridpoint
# ------------------------------------------------------------------------------
cpdef tuple compute_voronoi(
    double[:, :] X,
    double[:] wj,
    double[:, :] Y,
    int[:] idxgrid,
    str metric = "euclidean",
    int k = 2,
    object inv_cov = None
):
    """
    Compute the Voronoi tessellation of the data points X with respect to the
    grid points Y.

    Parameters:
    - X: Data matrix (N x D) of sample points
    - wj: Weights array (N,) for each sample point
    - Y: Data matrix (ngrid x D) of grid points
    - idxgrid: Indices array (ngrid,) of grid point indices in original dataset
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev", etc.)
    - k: Parameter for Minkowski distance (p value), default is 2
    - inv_cov: Optional inverse covariance matrix for mahalanobis distance

    Returns:
    - iminij: Assignment of each sample to a grid point (N,)
    - ni: Number of samples per Voronoi cell (ngrid,)
    - wi: Sum of weights per Voronoi cell (ngrid,)
    - ineigh: Closest sample index to each grid point (ngrid,)
    """
    cdef Py_ssize_t nsamples = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t ngrid = Y.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1] iminij = np.zeros(nsamples, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] ni = np.zeros(ngrid, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] wi = np.zeros(ngrid, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] ineigh = np.zeros(ngrid, dtype=np.int32)

    cdef double[:] dminij = np.full(nsamples, HUGE_VAL, dtype=np.float64)
    cdef double dij, dneigh
    cdef Py_ssize_t i, j

    # Assign each sample to closest grid point
    for i in range(ngrid):
        dneigh = HUGE_VAL
        for j in range(nsamples):
            dij = calculate_distance(metric, X[j, :], Y[i, :], k, inv_cov)
            if dij < dminij[j]:
                dminij[j] = dij
                iminij[j] = i
            if 0.0 < dij < dneigh:
                dneigh = dij
                ineigh[i] = j

    # Compute region sizes and weights
    for j in range(nsamples):
        ni[iminij[j]] += 1
        wi[iminij[j]] += wj[j]

    return iminij, ni, wi, ineigh

# ------------------------------------------------------------------------------
# Min-Max selection with flexible metrics
# ------------------------------------------------------------------------------
cpdef object select_grid_points(
    double[:, :] X,
    int ngrid,
    str metric = "euclidean",
    int k = 2,
    object inv_cov = None
):
    """
    Select 'ngrid' points from X (N x D) by the min–max algorithm, using one
    of multiple metrics.

    Parameters:
    - X: Data matrix (N x D)
    - ngrid: Number of points to select
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev",
              "cosine", "mahalanobis", "minkowski")
    - inv_cov: Optional inverse covariance matrix for mahalanobis distance,
               or 1x1 array with p parameter for minkowski distance
    - k: Parameter for Minkowski distance (p value), default is 2

    Returns:
    - idxgrid: Indices of selected grid points (ngrid,)
    - Y: Selected grid points matrix (ngrid x D)

    Raises:
    - ValueError: If metric is invalid, or if mahalanobis/minkowski metrics are used
                 without proper parameters, or if X is empty
    """
    # X is already a memory view that accepts C-contiguous arrays

    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]

    # Allocate result arrays
    cdef np.ndarray[np.int32_t, ndim=1] idxgrid = np.empty(ngrid, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] Y = np.empty((ngrid, D), dtype=np.float64)

    # Track the min distance of each point to any chosen grid point
    cdef np.ndarray[np.float64_t, ndim=1] dmin_ = np.empty(N, dtype=np.float64)

    # 1. Pick the first grid point randomly
    cdef int irandom = int(np.random.rand() * N)
    idxgrid[0] = irandom
    Y[0, :] = X[irandom, :]

    # Initialize dmin_ using the first chosen grid point
    cdef Py_ssize_t i, j
    cdef double max_min_dist, d

    for j in range(N):
        dmin_[j] = calculate_distance(metric, X[j, :], Y[0, :], k, inv_cov)

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
        Y[i, :] = X[jmax, :]

        # Update dmin_ with this newly selected point
        for j in range(N):
            d = calculate_distance(metric, X[j, :], Y[i, :], k, inv_cov)
            if d < dmin_[j]:
                dmin_[j] = d

    return idxgrid, Y
