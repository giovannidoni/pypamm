# cython: language_level=3
# Declarations for optimized operations in clustering module

cimport numpy as cnp
import numpy as np

# Declare the functions from _opx.pyx
cpdef object invmatrix(int D, double[:, ::1] M)
cpdef double trmatrix(int D, double[:, ::1] M)
cpdef double detmatrix(int D, double[:, ::1] M)
cpdef double logdet(int D, double[:, ::1] M)
cpdef double variance(int nsamples, int D, double[:, ::1] x, double[::1] weights)
cpdef object eigval(int D, double[:, ::1] AB)
cpdef double maxeigval(int D, double[:, ::1] AB)
cpdef int factorial(int n)
cpdef object pammrij(int D, double[::1] period, double[::1] ri, double[::1] rj)
cpdef double mahalanobis(int D, double[::1] period, double[::1] x, double[::1] y, double[:, ::1] Qinv)
cpdef double effdim(int D, double[:, ::1] Q)
cpdef object oracle(int D, double N, double[:, ::1] Q)
