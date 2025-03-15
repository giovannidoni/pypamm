# distance_metrics.pxd
# This file exposes the distance functions for use in other Cython modules

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

# Define the function pointer type for distance functions
ctypedef double (*dist_func_t)(double[::1], double[::1], double[:, ::1]) nogil

# Declare the C-level functions that will be exposed to other Cython modules
cdef double dist_euclidean(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil

cdef double dist_manhattan(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil

cdef double dist_chebyshev(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil

cdef double dist_cosine(
    double[::1] a,
    double[::1] b,
    double[:, ::1] unused
) except? -1 nogil

cdef double dist_mahalanobis(
    double[::1] a,
    double[::1] b,
    double[:, ::1] inv_cov
) except? -1 nogil

cdef double dist_minkowski(
    double[::1] a,
    double[::1] b,
    double[:, ::1] param
) except? -1 nogil

# Declare the internal function to get a distance function
cdef dist_func_t _get_distance_function(str metric) except *
