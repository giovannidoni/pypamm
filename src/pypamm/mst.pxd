# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from pypamm.distance_metrics cimport dist_func_t

# Helper functions for Union-Find
cdef int find_root(int v, int[:] parent) except? -1 nogil
cdef void union_sets(int v1, int v2, int[:] parent) noexcept nogil

# Public functions
cpdef np.ndarray[np.float64_t, ndim=2] build_mst(np.ndarray[np.float64_t, ndim=2] X, str metric=*)
