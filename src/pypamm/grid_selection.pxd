# grid_selection.pxd
# This file exposes the grid selection functions

import numpy as np
cimport numpy as np

cpdef object select_grid_points(
    double[::1, :] X,
    int ngrid,
    str metric = *,
    object inv_cov = *,
    double k = *
)

cpdef tuple compute_voronoi(
    double[::1, :] X,
    double[::1] wj,
    double[::1, :] Y,
    int[::1] idxgrid,
    str metric = *,
    object inv_cov = *,
    double k = *
)
