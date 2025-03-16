# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp
from scipy.linalg import eigh, inv


# -----------------------------------------------------------------------------
# 1) Compute bandwidth
# -----------------------------------------------------------------------------
cpdef compute_bandwidth(
    np.ndarray[np.float64_t, ndim=2] X,
    double constant_bandwidth=-1.0,
    double delta=1e-3,
    double tune=0.1,
    int max_iter=50,
    bint use_adaptive=True,
    double fpoints=None,
    double gspread=None
):
    """
    Computes bandwidth estimation with optional adaptivity.

    Parameters:
    - X: (N, D) Data points.
    - constant_bandwidth: If > 0, a fixed bandwidth is used.
    - delta: Convergence threshold.
    - tune: Initial step size for adaptation.
    - max_iter: Maximum iterations for tuning.
    - use_adaptive: Whether to use adaptive bandwidth computation.
    - fpoints: Fraction of points for bandwidth localization.
    - gspread: Spread factor for KDE scaling.

    Returns:
    - h: Global bandwidth (or fixed value if provided).
    - adaptive_bandwidths: (N,) Per-point adaptive bandwidths.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int i, iter_count
    cdef double prev_sigma2, new_sigma2, n_local

    # If a constant bandwidth is provided and adaptivity is disabled, return it directly
    if constant_bandwidth > 0 and not use_adaptive:
        return constant_bandwidth, np.full(N, constant_bandwidth, dtype=np.float64)

    # Ensure that only `fpoints` or `gspread` is used (mutual exclusivity)
    if fpoints is not None and gspread is not None:
        raise ValueError("Only one of `fpoints` or `gspread` can be set. Choose one.")

    # Default `fpoints` and `gspread` based on availability
    if fpoints is None and gspread is None:
        fpoints = 0.05  # Default to 5% of points if neither is set

    # Compute initial covariance estimate
    cdef np.ndarray[np.float64_t, ndim=2] Qi = np.cov(X, rowvar=False) + np.eye(D) * 1e-6  # Stability

    # **Adaptive Bandwidth Scaling:**
    if fpoints is not None:
        n_local = fpoints * N  # Fraction-based localization
        Hi = (4.0 / (D + 2.0)) ** (2.0 / (D + 4.0)) * n_local ** (-2.0 / (D + 4.0)) * Qi
    else:
        n_local = N  # Default full dataset
        Hi = gspread * Qi if gspread is not None else Qi  # Scale by `gspread`

    # Inverse bandwidth matrix
    Hiinv = np.linalg.inv(Hi)

    # Initialize per-point bandwidths
    cdef np.ndarray[np.float64_t, ndim=1] sigma2 = np.full(N, np.mean(np.diag(Qi)), dtype=np.float64)

    # **Apply Adaptive Bandwidth Refinement (if enabled)**
    if use_adaptive and fpoints is not None:
        for i in range(N):
            iter_count = 0
            prev_sigma2 = sigma2[i]

            while iter_count < max_iter:
                new_sigma2 = sigma2[i] + tune * (1 - alpha) if fpoints > 0.5 else sigma2[i] - tune * alpha

                # Compute local density estimate
                flocal = np.mean(np.linalg.norm(X - X[i], axis=1) < new_sigma2)

                # Adjust bandwidth to reach `fpoints * N` density
                if flocal > fpoints:
                    sigma2[i] -= tune * alpha / (2.0 ** iter_count)
                else:
                    sigma2[i] += tune * (1 - alpha) / (2.0 ** iter_count)

                iter_count += 1

                # Stop if convergence reached
                if abs(flocal - fpoints) < delta:
                    break

                # Detect oscillation (avoid infinite loops)
                if abs(new_sigma2 - prev_sigma2) < 1e-6:
                    break

                prev_sigma2 = new_sigma2

            # Ensure minimum bandwidth
            sigma2[i] = max(sigma2[i], np.min(np.linalg.norm(X - X[i], axis=1)) + 1e-6)

    # If `gspread` is used, scale bandwidth accordingly
    if gspread is not None:
        sigma2 *= gspread

    # Compute final bandwidth values
    adaptive_bandwidths = np.sqrt(sigma2)

    return np.mean(adaptive_bandwidths), adaptive_bandwidths


# -----------------------------------------------------------------------------
# 2) Gaussian Parameter Preparation
# -----------------------------------------------------------------------------
cpdef gauss_prepare(
    np.ndarray[np.float64_t, ndim=2] X,
    double alpha=0.5,
    bint adaptive=False,
    double constant_bandwidth=-1.0
):
    """
    Computes mean, covariance, and adaptive bandwidth matrix.

    Parameters:
    - X: (N, D) NumPy array of data points.
    - alpha: Adaptivity parameter for bandwidth.
    - adaptive: Whether to use adaptive bandwidth.
    - constant_bandwidth: If > 0, uses a fixed bandwidth instead.

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

    if N == 1:
        cov = np.eye(D, dtype=np.float64)  # Use identity covariance for single point
    else:
        cov = np.cov(X, rowvar=False)
        cov += np.eye(D) * 1e-6  # Ensure numerical stability

    # Compute inverse covariance matrix
    cdef np.ndarray[np.float64_t, ndim=2] inv_cov = np.linalg.inv(cov)

    # Compute eigenvalues
    cdef np.ndarray[np.float64_t, ndim=1] eigvals
    cdef np.ndarray[np.float64_t, ndim=2] eigvecs
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)  # Avoid zero or negative eigenvalues

    # Compute bandwidth matrix Hi
    cdef np.ndarray[np.float64_t, ndim=2] Hi
    cdef double h
    cdef np.ndarray[np.float64_t, ndim=1] adaptive_bandwidths

    if adaptive:
        h, adaptive_bandwidths = compute_bandwidth(X, alpha=alpha, constant_bandwidth=constant_bandwidth)
    else:
        h = constant_bandwidth if constant_bandwidth > 0 else 1.0  # Use fixed bandwidth

    Hi = np.diag(h * np.sqrt(eigvals))

    # Compute inverse bandwidth matrix Hiinv
    cdef np.ndarray[np.float64_t, ndim=2] Hiinv = np.linalg.inv(Hi)

    return mean, cov, inv_cov, eigvals, Hi, Hiinv


# -----------------------------------------------------------------------------
# 3) KDE Computation
# -----------------------------------------------------------------------------
cpdef compute_kde(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=2] grid,
    double alpha=0.5,
    bint adaptive=False,
    double constant_bandwidth=-1.0,
    object fpoints=None,
    object gspread=None
):
    """
    Computes Kernel Density Estimation (KDE) with optional adaptive bandwidth.

    Parameters:
    - X: (N, D) Data points.
    - grid: (G, D) Grid points where KDE is evaluated.
    - alpha: Adaptivity parameter.
    - adaptive: Whether to use adaptive bandwidth.
    - constant_bandwidth: If > 0, use a fixed bandwidth instead of adaptive estimation.
    - fpoints: Fraction of points for bandwidth localization.
    - gspread: Spread factor for KDE scaling.

    Returns:
    - density: (G,) KDE density values at each grid point.
    """
    cdef int N = X.shape[0]
    cdef int G = grid.shape[0]
    cdef int D = X.shape[1]
    cdef int i, j

    # Compute bandwidths (adaptive or fixed)
    h, adaptive_bandwidths = compute_bandwidth(X, alpha=alpha, constant_bandwidth=constant_bandwidth,
                                              fpoints=fpoints, gspread=gspread,
                                              use_adaptive=adaptive)

    # Allocate KDE result array
    cdef np.ndarray[np.float64_t, ndim=1] density = np.zeros(G, dtype=np.float64)

    # Compute KDE
    cdef double norm_factor
    cdef double dist_sq, weight
    cdef np.ndarray[np.float64_t, ndim=1] diff

    for i in range(G):
        for j in range(N):
            h_i = adaptive_bandwidths[j] if adaptive else h  # Use per-point bandwidth if adaptive
            norm_factor = (1 / (sqrt(2 * np.pi) * h_i) ** D)  # Normalize with adaptive bandwidth
            diff = (grid[i] - X[j]) / h_i
            dist_sq = np.dot(diff, diff)
            weight = np.exp(-0.5 * dist_sq)
            density[i] += weight * norm_factor

        density[i] /= N  # Normalize by number of points

    return density


# -----------------------------------------------------------------------------
# 4) KDE Cutoff Calculation
# -----------------------------------------------------------------------------
cpdef double kde_cutoff(int D, double alpha=0.5):
    """
    Computes KDE cutoff (`kdecut2`) for the given dimensionality.

    Parameters:
    - D: Dimensionality of data.
    - alpha: Adaptivity parameter.

    Returns:
    - kdecut2: KDE squared cutoff.
    """
    return 9.0 * (sqrt(D) + alpha) ** 2


# -----------------------------------------------------------------------------
# 5) KDE Bootstrap Error Estimation
# -----------------------------------------------------------------------------
cpdef kde_bootstrap_error(
    np.ndarray[np.float64_t, ndim=2] X,
    int n_bootstrap,
    double alpha=0.5,
    bint adaptive=False,
    double constant_bandwidth=-1.0
):
    """
    Estimates KDE statistical error using bootstrap resampling.

    Parameters:
    - X: (N, D) Data points.
    - n_bootstrap: Number of bootstrap runs.
    - alpha: Adaptivity parameter for bandwidth.
    - adaptive: Whether to use adaptive bandwidth.
    - constant_bandwidth: If > 0, uses a fixed bandwidth instead.

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
        boot_kdes[b] = compute_kde(boot_sample, grid, alpha, adaptive, constant_bandwidth)

    cdef np.ndarray[np.float64_t, ndim=1] mean_kde = np.mean(boot_kdes, axis=0)
    cdef np.ndarray[np.float64_t, ndim=1] std_kde = np.std(boot_kdes, axis=0)

    return mean_kde, std_kde


# -----------------------------------------------------------------------------
# 6) KDE Output Storage
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
