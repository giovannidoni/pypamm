# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libc.math cimport HUGE_VAL
from pypamm.lib.distance cimport (
    calculate_distance
)
import numpy as np
cimport numpy as np

# ------------------------------------------------------------------------------
# Build a neighbor list from Voronoi
# ------------------------------------------------------------------------------
cpdef tuple get_voronoi_neighbour_list(
    int nsamples,
    int ngrid,
    np.ndarray[int, ndim=1] ni,         # shape: (ngrid,)
    np.ndarray[int, ndim=1] iminij      # shape: (nsamples,)
):
    """
    Build a neighbor list for each Voronoi cell.

    For each Voronoi center (grid point), this function collects the indices
    of the sample points that belong to it, and returns both the neighbor list
    and a pointer list for fast access.

    Parameters:
    - nsamples: Total number of sample points
    - ngrid: Number of Voronoi cells (grid points)
    - ni: Array of size (ngrid,) with number of samples assigned to each Voronoi cell
    - iminij: Array of size (nsamples,) assigning each sample to a Voronoi cell

    Returns:
    - pnlist: Pointer array (ngrid + 1,) to segment boundaries in the neighbor list.
              For Voronoi cell j, the sample indices are in nlist[pnlist[j]:pnlist[j+1]]
    - nlist: Flattened neighbor list (nsamples,) with indices of samples grouped by Voronoi cell
    """
    cdef int i, j
    cdef np.ndarray[int, ndim=1] pnlist = np.zeros(ngrid + 1, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] nlist = np.zeros(nsamples, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] tmpnidx = np.zeros(ngrid, dtype=np.int32)

    # Build pnlist and tmpnidx
    pnlist[0] = 0
    for i in range(ngrid):
        pnlist[i + 1] = pnlist[i] + ni[i]
        tmpnidx[i] = pnlist[i]

    # Fill neighbor list
    for j in range(nsamples):
        i = iminij[j]
        nlist[tmpnidx[i]] = j
        tmpnidx[i] += 1

    return pnlist, nlist


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
