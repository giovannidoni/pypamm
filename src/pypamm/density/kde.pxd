# cython: language_level=3
cimport numpy as np
from libc.math cimport exp, sqrt

cpdef gauss_prepare(np.ndarray[np.float64_t, ndim=2] X)
cpdef compute_kde(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] grid, double bandwidth)
cpdef double kde_cutoff(int D)
cpdef kde_bootstrap_error(np.ndarray[np.float64_t, ndim=2] X, int n_bootstrap, double bandwidth)
cpdef kde_output(np.ndarray[np.float64_t, ndim=1] density, np.ndarray[np.float64_t, ndim=1] std_kde)
