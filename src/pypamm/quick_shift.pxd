import numpy as np
cimport numpy as np
from pypamm.distance_metrics cimport dist_func_t

# Declare the public API
cpdef int qs_next(int ngrid, int idx, int idxn, double lambda_, double[:] probnmm, double[:, :] distmm)
