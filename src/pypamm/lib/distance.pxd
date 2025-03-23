# distance_metrics.pxd
# This file exposes the distance functions for use in other Cython modules

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

# Declare the C-level functions that will be exposed to other Cython modules
cdef double dist_euclidean(
    double[:] a,
    double[:] b,
) except? -1 nogil

cdef double dist_manhattan(
    double[:] a,
    double[:] b,
) except? -1 nogil

cdef double dist_chebyshev(
    double[:] a,
    double[:] b,
) except? -1 nogil

cdef double dist_cosine(
    double[:] a,
    double[:] b,
) except? -1 nogil

cdef double dist_mahalanobis(
    double[:] a,
    double[:] b,
    double[:, :] inv_cov
) except? -1 nogil

cdef double dist_minkowski(
    double[:] a,
    double[:] b,
    int k
) except? -1 nogil

# Declare the internal function to get a distance function
cdef double calculate_distance(str metric, double[:] a, double[:] b, int k = *, object inv_cov = *) except *

# Declare a function to get a distance function callable in python
cpdef double py_calculate_distance(str metric, double[:] a, double[:] b, int k = *, object inv_cov = *) except *

# Declare function to calculate distance matrix
cpdef tuple compute_pairwise_distances(double[:, :] a, str metric = *, int k = *, object inv_cov = *)
