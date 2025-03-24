# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, pow, exp
from math import factorial as py_factorial
from math import floor, ceil


# Initialize NumPy (required once per extension)
cnp.import_array()

# Define numpy types for convenience
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

########## invmatrix ##########
cpdef object invmatrix(double[:, ::1] M):
    """
    Here we just return the inverted matrix as a new NumPy array.
    """
    # In Fortran, IM was the output array. In Python, we just return it.
    cdef cnp.ndarray[DTYPE_t, ndim=2] IM = np.linalg.inv(M)
    return IM

########## trmatrix ##########
cpdef double trmatrix(double[:, ::1] M):
    """
    Return the trace of a square matrix.
    """
    return np.trace(M)

########## detmatrix ##########
cpdef double detmatrix(double[:, ::1] M):
    """
    Return the determinant of a square matrix.
    """
    return np.linalg.det(M)

########## logdet ##########
cpdef double logdet(double[:, ::1] M):
    """
    Return the log determinant of a square matrix.
    """
    cdef double sign, ld
    sign, ld = np.linalg.slogdet(M)
    # For a real matrix, sign can be 1 or -1. In Fortran, if det < 0, it returns 0.
    # Here we just return ld directly (log|det|).
    return ld

########## eigval ##########
cpdef object eigval(double[:, ::1] AB):
    """
    In Python, we just return the eigenvalues of AB as a NumPy array.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] WR = np.linalg.eigvals(AB)
    return WR

########## maxeigval ##########
cpdef double maxeigval(double[:, ::1] AB):
    """
    Return the max real eigenvalue.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] vals = np.linalg.eigvals(AB)
    return vals.max()

########## factorial ##########
cpdef int factorial(int n):
    """
    Fortran FUNCTION factorial(n).
    """
    return py_factorial(n)

########## effdim ##########
cpdef double effdim(double[:, ::1] Q):
    """
    Uses the eigenvalue-based formula with pk * log(pk).
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] vals = np.linalg.eigvals(Q)
    cdef double s = np.sum(vals)
    if s == 0:
        return 0.0
    cdef cnp.ndarray[DTYPE_t, ndim=1] pk = vals / s

    cdef double sum_plogp = 0.0
    cdef int i, n = pk.shape[0]
    for i in range(n):
        if pk[i] > 0.0:
            sum_plogp += pk[i] * log(pk[i])

    # effdim = exp(- sum of pk log pk)
    return np.exp(-sum_plogp)

########## oracle ##########
cpdef object oracle(double[:, ::1] Q, double N):
    """
    Applies a shrinkage transform in-place on Q.
    """
    cdef Py_ssize_t D = Q.shape[1]
    cdef double trQ = 0.0
    cdef double trQ2 = 0.0
    cdef int i, j

    # Sum of diagonal
    for i in range(D):
        trQ += Q[i, i]
    trQ2 = trQ * trQ

    # Sum of squares of diagonal
    cdef double sum_diag_sq = 0.0
    for i in range(D):
        sum_diag_sq += Q[i, i] * Q[i, i]

    # The fraction
    #   phi = ((1 - 2/D) * sum_diag_sq + trQ^2) /
    #         ((N + 1 - 2/D) * sum_diag_sq - trQ^2/D)
    cdef double numerator = (1.0 - 2.0 / D) * sum_diag_sq + trQ2
    cdef double denominator = (N + 1.0 - 2.0 / D) * sum_diag_sq - (trQ2 / D)

    cdef double phi
    if denominator == 0:
        phi = 1.0
    else:
        phi = numerator / denominator

    # clamp phi to [0, 1]
    if phi < 0.0:
        phi = 0.0
    elif phi > 1.0:
        phi = 1.0

    cdef double rho = phi
    cdef double alpha = 1.0 - rho
    cdef double avg_diag = trQ / D

    # Scale off-diagonal by alpha, add rho * avg_diag on diagonal
    for i in range(D):
        for j in range(D):
            if i == j:
                Q[i, j] = alpha * Q[i, j] + rho * avg_diag
            else:
                Q[i, j] = alpha * Q[i, j]

    # No return — modifies Q in place, as a Fortran SUBROUTINE would.
    return Q

########## weighted covariance ##########
cpdef object wcovariance(
    double[:, ::1] x,        # shape (N, D)
    double[::1] w,         # shape (N,)
    double wnorm
):
    """
    Calculate weighted covariance.
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t D = x.shape[1]
    cdef int n = N
    cdef int d = D

    cdef np.ndarray[np.float64_t, ndim=1] xm = np.zeros(D, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] xxm = np.zeros((N, D), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] xxmw = np.zeros((N, D), dtype=np.float64)

    cdef Py_ssize_t i, j

    # Compute weighted mean
    for j in range(D):
        for i in range(N):
            xm[j] += x[i, j] * w[i]
        xm[j] /= wnorm

    # Compute differences and weighted differences
    for i in range(N):
        for j in range(D):
            xxm[i, j] = x[i, j] - xm[j]
            xxmw[i, j] = xxm[i, j] * w[i] / wnorm

    # Call DGEMM: Q = xxm.T @ xxmw
    cdef np.ndarray[np.float64_t, ndim=2] Q = np.dot(xxm.T, xxmw)

    # Denominator: 1 - sum((w / wnorm)^2)
    cdef double correction = 0.0
    for i in range(N):
        correction += (w[i] / wnorm) ** 2

    for i in range(D):
        for j in range(D):
            Q[i, j] /= (1.0 - correction)

    return Q


########## compute localised weight ##########
cpdef tuple compute_localization(
    double[:, ::1] x,      # shape (N, D)
    double[::1] y,       # shape (D,)
    double[::1] w,       # shape (N,)
    double sigma
):
    """
    Calculate weighted covariance.
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t D = x.shape[1]
    cdef Py_ssize_t i, j
    cdef double diff, dist2
    cdef np.ndarray[np.float64_t, ndim=1] wl = np.empty(N, dtype=np.float64)
    cdef double num = 0.0

    for i in range(N):
        dist2 = 0.0
        for j in range(D):
            diff = x[i, j] - y[j]
            dist2 += diff * diff
        wl[i] = exp(-0.5 * dist2 / sigma) * w[i]
        num += wl[i]

    return wl, num
