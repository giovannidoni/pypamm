# cython: language_level=3
cimport numpy as np
from libc.math cimport exp, sqrt

cpdef compute_bandwidth(np.ndarray[np.float64_t, ndim=2] X, double alpha=*, double constant_bandwidth=*, double delta=*, double tune=*, int max_iter=*, bint use_adaptive=*, double fpoints=*, double gspread=*)
cpdef gauss_prepare(np.ndarray[np.float64_t, ndim=2] X, double alpha=*, bint adaptive=*, double constant_bandwidth=*, double fpoints=*, double gspread=*)
cpdef compute_kde(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] grid, double alpha=*, bint adaptive=*, double constant_bandwidth=*, double fpoints=*, double gspread=*)
cpdef double kde_cutoff(int D, double alpha=*)
cpdef kde_bootstrap_error(np.ndarray[np.float64_t, ndim=2] X, int n_bootstrap, double alpha=*, bint adaptive=*, double constant_bandwidth=*, double fpoints=*, double gspread=*)
cpdef kde_output(np.ndarray[np.float64_t, ndim=1] density, np.ndarray[np.float64_t, ndim=1] std_kde)
