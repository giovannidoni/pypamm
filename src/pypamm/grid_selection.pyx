# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

# Import distance functions from the distance_metrics module
from pypamm.distance_metrics cimport (
    calculate_distance
)
from libc.math cimport HUGE_VAL

# ------------------------------------------------------------------------------
# Associate each point to the closet gridpoint
# ------------------------------------------------------------------------------
cpdef tuple compute_voronoi(
    double[::1, :] X,
    double[::1] wj,
    double[::1, :] Y,
    int[::1] idxgrid,
    str metric = "euclidean",
    object inv_cov = None,
    double k = 2.0
):
    """
    Assigns each sample in X to the closest grid point in Y (Voronoi assignment).

    Arguments:
        X: Sample matrix (N x D)
        wj: Weights for each sample (N,)
        Y: Grid points (ngrid x D)
        idxgrid: Indices of selected grid points (ngrid,)
        metric: Distance metric (e.g. "euclidean")

    Returns:
        iminij: np.ndarray[int32_t, ndim=1] - attribution of each sample
        ni:     np.ndarray[int32_t, ndim=1] - number of samples per Voronoi cell
        wi:     np.ndarray[float64, ndim=1] - sum of weights per Voronoi cell
        ineigh: np.ndarray[int32_t, ndim=1] - closest sample to each grid point
    """
    cdef Py_ssize_t nsamples = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t ngrid = Y.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1] iminij = np.zeros(nsamples, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] ni = np.zeros(ngrid, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] wi = np.zeros(ngrid, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] ineigh = np.zeros(ngrid, dtype=np.int32)

    cdef double[::1] dminij = np.full(nsamples, HUGE_VAL, dtype=np.float64)
    cdef double dij, dneigh
    cdef Py_ssize_t i, j

    # Assign each sample to closest grid point
    for i in range(ngrid):
        dneigh = HUGE_VAL
        for j in range(nsamples):
            dij = calculate_distance(metric, X[j, :], Y[i, :], inv_cov, k)
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
    double[::1, :] X,
    int ngrid,
    str metric = "euclidean",
    object inv_cov = None,
    double k = 2.0
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
        dmin_[j] = calculate_distance(metric, X_arr[j, :], Y[0, :], inv_cov, k)

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
            d = calculate_distance(metric, X_arr[j, :], Y[i, :], inv_cov, k)
            if d < dmin_[j]:
                dmin_[j] = d

    return idxgrid, Y
