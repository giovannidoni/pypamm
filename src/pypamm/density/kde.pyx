# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, M_PI

# -----------------------------------------------------------------------------
# 1️) Gaussian Parameter Preparation
# -----------------------------------------------------------------------------
cpdef gauss_prepare(np.ndarray[np.float64_t, ndim=2] X):
    """
    Computes mean and covariance for Gaussian KDE.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    
    cdef np.ndarray[np.float64_t, ndim=1] mean = np.mean(X, axis=0)
    cdef np.ndarray[np.float64_t, ndim=2] cov = np.cov(X, rowvar=False)

    return mean, cov

# -----------------------------------------------------------------------------
# 2️) KDE Computation
# -----------------------------------------------------------------------------
cpdef compute_kde(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] grid, double bandwidth):
    """
    Computes Kernel Density Estimation (KDE) on given grid points.

    Parameters:
    - X: (N, D) Data points.
    - grid: (G, D) Grid points where KDE is evaluated.
    - bandwidth: Bandwidth parameter for KDE.

    Returns:
    - density: (G,) KDE density values at each grid point.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int G = grid.shape[0]

    cdef np.ndarray[np.float64_t, ndim=1] density = np.zeros(G, dtype=np.float64)
    cdef double norm_factor = (1 / (sqrt(2 * M_PI) * bandwidth)) ** D
    
    cdef int i, j
    cdef double dist_sq, weight

    for i in range(G):
        for j in range(N):
            dist_sq = np.sum((grid[i] - X[j]) ** 2)
            weight = exp(-0.5 * dist_sq / (bandwidth ** 2))
            density[i] += weight
        
        density[i] *= norm_factor / N

    return density

# -----------------------------------------------------------------------------
# 3️) KDE Cutoff Calculation
# -----------------------------------------------------------------------------
cpdef double kde_cutoff(int D):
    """
    Computes KDE cutoff (`kdecut2`) for the given dimensionality.

    Returns:
    - kdecut2: KDE squared cutoff.
    """
    return 9.0 * (sqrt(D) + 1.0) ** 2

# -----------------------------------------------------------------------------
# 4️) KDE Bootstrap Error Estimation
# -----------------------------------------------------------------------------
cpdef kde_bootstrap_error(np.ndarray[np.float64_t, ndim=2] X, int n_bootstrap, double bandwidth):
    """
    Estimates KDE statistical error using bootstrap resampling.

    Parameters:
    - X: (N, D) Data points.
    - n_bootstrap: Number of bootstrap runs.
    - bandwidth: Bandwidth parameter.

    Returns:
    - mean_kde: Mean KDE values.
    - std_kde: Standard deviation of KDE estimates.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] grid = X.copy()  # Evaluate KDE on original points

    cdef np.ndarray[np.float64_t, ndim=2] boot_kdes = np.zeros((n_bootstrap, N), dtype=np.float64)
    
    cdef int b, i
    for b in range(n_bootstrap):
        boot_sample = X[np.random.choice(N, N, replace=True)]
        boot_kdes[b] = compute_kde(boot_sample, grid, bandwidth)
    
    cdef np.ndarray[np.float64_t, ndim=1] mean_kde = np.mean(boot_kdes, axis=0)
    cdef np.ndarray[np.float64_t, ndim=1] std_kde = np.std(boot_kdes, axis=0)

    return mean_kde, std_kde

# -----------------------------------------------------------------------------
# 5️) KDE Output Storage
# -----------------------------------------------------------------------------
cpdef kde_output(np.ndarray[np.float64_t, ndim=1] density, np.ndarray[np.float64_t, ndim=1] std_kde):
    """
    Stores KDE outputs.

    Parameters:
    - density: (G,) KDE density values.
    - std_kde: (G,) KDE standard deviations.

    Returns:
    - prb: KDE density values.
    - aer: Absolute errors on KDE.
    - rer: Relative errors on KDE.
    """
    cdef np.ndarray[np.float64_t, ndim=1] prb = density
    cdef np.ndarray[np.float64_t, ndim=1] aer = std_kde
    cdef np.ndarray[np.float64_t, ndim=1] rer = std_kde / (density + 1e-8)  # Avoid division by zero

    return prb, aer, rer
    