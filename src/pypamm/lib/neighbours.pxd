# This file exposes the neighbour function

import numpy as np
cimport numpy as np


cpdef tuple get_voronoi_neighbour_list(
    int nsamples,
    int ngrid,
    np.ndarray[int, ndim=1] ni,
    np.ndarray[int, ndim=1] iminij
)

cpdef tuple compute_voronoi(
    double[:, :] X,
    double[:] wj,
    double[:, :] Y,
    int[:] idxgrid,
    str metric = *,
    int k = *,
    object inv_cov = *
)
