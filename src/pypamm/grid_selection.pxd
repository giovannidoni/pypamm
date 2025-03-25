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
