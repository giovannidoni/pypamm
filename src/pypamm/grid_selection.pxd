# grid_selection.pxd
# This file exposes the distance functions for use in other Cython modules

import numpy as np
cimport numpy as np

# Define the function pointer type for distance functions
ctypedef double (*dist_func_t)(double[::1], double[::1], double[:, ::1]) nogil

# Declare the distance functions with explicit exception values
cdef double dist_euclidean(double[::1] a, double[::1] b, double[:, ::1] unused) except? -1 nogil
cdef double dist_manhattan(double[::1] a, double[::1] b, double[:, ::1] unused) except? -1 nogil
cdef double dist_chebyshev(double[::1] a, double[::1] b, double[:, ::1] unused) except? -1 nogil
cdef double dist_cosine(double[::1] a, double[::1] b, double[:, ::1] unused) except? -1 nogil
cdef double dist_mahalanobis(double[::1] a, double[::1] b, double[:, ::1] inv_cov) except? -1 nogil
cdef double dist_minkowski(double[::1] a, double[::1] b, double[:, ::1] param) except? -1 nogil 