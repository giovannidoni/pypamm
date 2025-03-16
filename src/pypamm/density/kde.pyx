# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp
from scipy.linalg import eigh, inv


# -----------------------------------------------------------------------------
# 1️) Gaussian Parameter Preparation
# -----------------------------------------------------------------------------
cpdef gauss_prepare(np.ndarray[np.float64_t, ndim=2] X):
    """
    Computes mean, covariance, bandwidth matrix (Hi), and inverse bandwidth (Hiinv).

    Parameters:
    - X: (N, D) NumPy array of data points.

    Returns:
    - mean: (D,) Mean vector.
    - cov: (D x D) Covariance matrix.
    - inv_cov: (D x D) Inverted covariance matrix.
    - eigvals: (D,) Eigenvalues of the covariance matrix.
    - Hi: (D x D) Bandwidth matrix.
    - Hiinv: (D x D) Inverse bandwidth matrix.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]

    # Compute mean
    cdef np.ndarray[np.float64_t, ndim=1] mean = np.mean(X, axis=0)

    # Compute covariance matrix
    cdef np.ndarray[np.float64_t, ndim=2] cov

    # Handle edge case: single point
    if N == 1:
        # For a single point, use identity covariance
        cov = np.eye(D, dtype=np.float64)
    elif D == 1:
        # For 1D data, compute variance manually
        cov = np.array([[np.var(X.flatten())]], dtype=np.float64)
    else:
        # Normal case: compute covariance
        try:
            cov = np.cov(X, rowvar=False)

            # Handle potential numerical issues
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                cov = np.eye(D, dtype=np.float64)
        except:
            # Fallback to identity matrix if covariance computation fails
            cov = np.eye(D, dtype=np.float64)

    # Compute inverse covariance matrix
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov = np.linalg.inv(cov)  # Stable inversion

    # Compute eigenvalues and eigenvectors of covariance
    cdef np.ndarray[np.float64_t, ndim=1] eigvals
    cdef np.ndarray[np.float64_t, ndim=2] eigvecs
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Ensure positive eigenvalues (numerical stability)
    eigvals = np.maximum(eigvals, 1e-10)

    # Compute bandwidth matrix Hi (scaled eigenvalues)
    cdef np.ndarray[np.float64_t, ndim=2] Hi = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        Hi[i, i] = sqrt(eigvals[i])  # Scale bandwidth by sqrt of eigenvalues

    # Compute inverse bandwidth matrix Hiinv
    cdef np.ndarray[np.float64_t, ndim=2] Hiinv = np.linalg.inv(Hi)

    return mean, cov, inv_cov, eigvals, Hi, Hiinv


# -----------------------------------------------------------------------------
# 2️) KDE Computation
# -----------------------------------------------------------------------------
cpdef compute_kde(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] grid, double bandwidth):
    """
    Computes Kernel Density Estimation (KDE) with covariance-adaptive bandwidth.

    Parameters:
    - X: (N, D) Data points.
    - grid: (G, D) Grid points where KDE is evaluated.
    - bandwidth: Scaling factor for the covariance-based bandwidth.

    Returns:
    - density: (G,) KDE density values at each grid point.
    """
    # Ensure arrays are float64
    X = np.asarray(X, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int G = grid.shape[0]

    # Compute covariance-based bandwidth
    mean, cov, inv_cov, eigvals, Hi, Hiinv = gauss_prepare(X)

    # Scale covariance by bandwidth parameter
    cov = cov * bandwidth
    inv_cov = inv_cov / bandwidth
    Hi = Hi * sqrt(bandwidth)
    Hiinv = Hiinv / sqrt(bandwidth)

    # Compute Mahalanobis KDE
    cdef np.ndarray[np.float64_t, ndim=1] density = np.zeros(G, dtype=np.float64)
    cdef double norm_factor = (1 / (sqrt(2 * np.pi))) ** D

    cdef int i, j
    cdef double dist_sq, weight
    cdef np.ndarray[np.float64_t, ndim=1] diff

    for i in range(G):
        for j in range(N):
            # Compute Mahalanobis distance with bandwidth scaling
            diff = np.dot(Hiinv, (grid[i] - X[j]))
            dist_sq = np.dot(diff, diff)  # Mahalanobis distance squared
            weight = np.exp(-0.5 * dist_sq)
            density[i] += weight

        density[i] *= norm_factor / (N * np.linalg.det(Hi))  # Adjust normalization

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
