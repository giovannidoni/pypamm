# grid_selection.pxd
# This file exposes the grid selection functions

import numpy as np
cimport numpy as np

# Declare the select_grid_points function
cpdef object select_grid_points(
    double[:, :] X,
    int ngrid,
    str metric = *,
    object inv_cov = *
)
