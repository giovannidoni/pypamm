# cython: language_level=3
# Declarations for optimized operations in clustering module

cimport numpy as cnp
import numpy as np

# Declare the functions from _opx.pyx
cpdef object invmatrix(double[:, ::1] M)
cpdef double trmatrix(double[:, ::1] M)
cpdef double detmatrix(double[:, ::1] M)
cpdef double logdet(double[:, ::1] M)
cpdef object eigval(double[:, ::1] AB)
cpdef double maxeigval(double[:, ::1] AB)
cpdef int factorial(int n)
cpdef double effdim(double[:, ::1] Q)
cpdef object oracle(double[:, ::1] Q, double N)
cpdef object wcovariance(double[:, ::1] x, double[::1] weights, double wnorm)
