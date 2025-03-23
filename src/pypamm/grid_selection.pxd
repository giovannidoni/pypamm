# grid_selection.pxd
# This file exposes the grid selection functions

import numpy as np
cimport numpy as np

cpdef object select_grid_points(
    double[:, :] X,
    int ngrid,
    str metric = *,
    int k = *,
    object inv_cov = *
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
