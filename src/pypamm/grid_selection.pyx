# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

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

    Args:
        X: (N x D) array of data points
        wj: (ngrid) array of weights
        Y: (ngrid x D) array of grid points
        idxgrid: (ngrid) array of grid point indices
        metric: distance metric to use
        inv_cov: inverse covariance matrix (for Mahalanobis)
        k: parameter for Minkowski distance (p value)

    Returns:
        Tuple of (iminij, ni, wi, ineigh)
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
    of multiple metrics:
    - euclidean
    - manhattan
    - chebyshev
    - cosine
    - mahalanobis (requires inv_cov matrix)
    - minkowski (requires k parameter)

    Args:
        X: (N x D) array of data points
        ngrid: number of points to select
        metric: distance metric to use
        inv_cov: inverse covariance matrix (for Mahalanobis)
        k: parameter for Minkowski distance (p value)

    Returns:
        (idxgrid, Y): indices of selected points and the corresponding data
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
