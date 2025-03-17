# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport log, sqrt
from math import factorial as py_factorial
from math import floor, ceil

# Initialize NumPy (required once per extension)
cnp.import_array()

# Define numpy types for convenience
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

########## invmatrix ##########
cpdef object invmatrix(int D, double[:, ::1] M):
    """
    Fortran SUBROUTINE invmatrix(D,M,IM)
    Here we just return the inverted matrix as a new NumPy array.
    """
    # In Fortran, IM was the output array. In Python, we just return it.
    cdef cnp.ndarray[DTYPE_t, ndim=2] IM = np.linalg.inv(M)
    return IM

########## trmatrix ##########
cpdef double trmatrix(int D, double[:, ::1] M):
    """
    Fortran FUNCTION trmatrix(D,M)
    Return the trace of a square matrix.
    """
    return np.trace(M)

########## detmatrix ##########
cpdef double detmatrix(int D, double[:, ::1] M):
    """
    Fortran FUNCTION detmatrix(D,M)
    Return the determinant of a square matrix.
    """
    return np.linalg.det(M)

########## logdet ##########
cpdef double logdet(int D, double[:, ::1] M):
    """
    Fortran FUNCTION logdet(D,M)
    Return the log determinant of a square matrix.
    """
    cdef double sign, ld
    sign, ld = np.linalg.slogdet(M)
    # For a real matrix, sign can be 1 or -1. In Fortran, if det < 0, it returns 0.
    # Here we just return ld directly (log|det|).
    return ld

########## variance ##########
cpdef double variance(int nsamples, int D, double[:, ::1] x, double[::1] weights):
    """
    Fortran FUNCTION variance(nsamples,D,x,weights).
    Weighted variance calculation.

    x has shape (D, nsamples).
    weights has shape (nsamples,).
    """
    # Follows the formula:
    # variance = wsum/( (sum(weights))^2 - sum(weights^2) ) * sum( weights * (NORM2(xtmp,1))^2 )
    # where xtmp(i,:) = x(i,:)-xm(i).
    # The Fortran code also prints "Mean", but we'll skip that here.
    cdef double wsum = np.sum(weights)
    # Weighted mean of each row
    # shape (D,) for xm
    cdef cnp.ndarray[DTYPE_t, ndim=1] xm = np.zeros(D, dtype=np.float64)
    cdef int i, j

    # Calculate weighted mean manually
    for i in range(D):
        for j in range(nsamples):
            xm[i] += x[i, j] * weights[j]
        xm[i] /= wsum

    # Build xtmp = x - xm row-wise
    # In Fortran: xtmp(i,:) = x(i,:) - xm(i)
    cdef cnp.ndarray[DTYPE_t, ndim=2] xtmp = np.zeros((D, nsamples), dtype=np.float64)
    for i in range(D):
        for j in range(nsamples):
            xtmp[i, j] = x[i, j] - xm[i]

    # Sum of weights^2
    cdef double sum_w2 = 0.0
    for j in range(nsamples):
        sum_w2 += weights[j] * weights[j]

    # Weighted sum of squared norms
    # We interpret "NORM2(xtmp,1)" as the 2-norm of each row (?),
    # but the Fortran code seems ambiguous.
    # Typically, you'd want the 2-norm of each sample vector => columns.
    # Let's do column-based norms (since Fortran indexing is x(i,sample)):
    cdef double wsum_norm = 0
    cdef double tmp_norm
    for j in range(nsamples):
        # 2-norm of xtmp[:, j]
        tmp_norm = 0
        for i in range(D):
            tmp_norm += xtmp[i, j] * xtmp[i, j]
        wsum_norm += weights[j] * tmp_norm  # (norm)^2

    # Full factor
    cdef double denominator = (wsum*wsum - sum_w2)
    if denominator == 0:
        # to avoid dividing by zero
        return 0.0
    cdef double varval = wsum / denominator * wsum_norm

    return varval

########## eigval ##########
cpdef object eigval(int D, double[:, ::1] AB):
    """
    Fortran SUBROUTINE eigval(AB,D,WR)
    In Python, we just return the eigenvalues of AB as a NumPy array.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] WR = np.linalg.eigvals(AB)
    return WR

########## maxeigval ##########
cpdef double maxeigval(int D, double[:, ::1] AB):
    """
    Fortran FUNCTION maxeigval(AB,D).
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

########## pammrij ##########
cpdef object pammrij(int D,
            double[::1] period,
            double[::1] ri,
            double[::1] rj):
    """
    Fortran SUBROUTINE pammrij(D,period,ri,rj,rij).
    Returns the minimum-image difference vector rij.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] rij = np.zeros(D, dtype=np.float64)
    cdef int k
    for k in range(D):
        rij[k] = ri[k] - rj[k]
        if period[k] > 0.0:
            # scale
            rij[k] /= period[k]
            # minimum image
            # Fortran's DNINT is "round to nearest integer"
            rij[k] -= round(rij[k])
            # scale back
            rij[k] *= period[k]
    return rij

########## mahalanobis ##########
cpdef double mahalanobis(int D,
                double[::1] period,
                double[::1] x,
                double[::1] y,
                double[:, ::1] Qinv):
    """
    Fortran FUNCTION mahalanobis(D,period,x,y,Qinv).
    Return the Mahalanobis distance.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] dv = pammrij(D, period, x, y)
    # dv @ Qinv => shape (D,)
    cdef cnp.ndarray[DTYPE_t, ndim=1] tmp = np.zeros(D, dtype=np.float64)
    cdef int i, j

    # Manual matrix-vector multiplication
    for i in range(D):
        for j in range(D):
            tmp[i] += dv[j] * Qinv[j, i]

    cdef double dist = 0.0
    for i in range(D):
        dist += dv[i] * tmp[i]
    return dist

########## effdim ##########
cpdef double effdim(int D, double[:, ::1] Q):
    """
    Fortran FUNCTION effdim(D,Q).
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
cpdef object oracle(int D, double N, double[:, ::1] Q):
    """
    Fortran SUBROUTINE oracle(D,N,Q).
    Applies a shrinkage transform in-place on Q.
    """
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
    return None
