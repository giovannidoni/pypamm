# grid_selection.pxd
# This file exposes the grid selection functions

import numpy as np
cimport numpy as np

# Import the distance function type from distance_metrics
from pypamm.distance_metrics cimport dist_func_t

# Declare the select_grid_points function
cpdef object select_grid_points(
    object X,
    int ngrid,
    str metric = *,
    object inv_cov = *
)
