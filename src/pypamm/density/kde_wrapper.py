"""
Python wrapper for the KDE Cython module.
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Flag to track which implementation is being used
_using_cython = False

# Import the Cython implementation
try:
    from pypamm.density.kde import (
        compute_kde as _compute_kde_cython,
    )
    from pypamm.density.kde import (
        gauss_prepare as _gauss_prepare_cython,
    )
    from pypamm.density.kde import (
        kde_bootstrap_error as _kde_bootstrap_error,
    )
    from pypamm.density.kde import (
        kde_cutoff as _kde_cutoff,
    )
    from pypamm.density.kde import (
        kde_output as _kde_output,
    )

    _using_cython = True
except ImportError:
    # Fallback implementations for testing or when Cython module is not available
    _using_cython = False


# Python fallback implementations
def _gauss_prepare_python(
    X: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64],  # mean
    NDArray[np.float64],  # cov
    NDArray[np.float64],  # inv_cov
    NDArray[np.float64],  # eigvals
    NDArray[np.float64],  # Hi
    NDArray[np.float64],  # Hiinv
]:
    """Fallback implementation of gauss_prepare with full output like Cython version."""
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Compute additional values for compatibility with Cython version
    inv_cov = np.linalg.inv(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Compute bandwidth matrix Hi (scaled eigenvalues)
    D = X.shape[1]
    Hi = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        Hi[i, i] = np.sqrt(eigvals[i])  # Scale bandwidth by sqrt of eigenvalues

    # Compute inverse bandwidth matrix
    Hiinv = np.linalg.inv(Hi)

    return mean, cov, inv_cov, eigvals, Hi, Hiinv


def _compute_kde_python(X: NDArray[np.float64], grid: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    """Fallback implementation of compute_kde."""
    N, D = X.shape
    G = grid.shape[0]
    density = np.zeros(G, dtype=np.float64)
    norm_factor = (1 / (np.sqrt(2 * np.pi) * bandwidth)) ** D

    for i in range(G):
        for j in range(N):
            dist_sq = np.sum((grid[i] - X[j]) ** 2)
            weight = np.exp(-0.5 * dist_sq / (bandwidth**2))
            density[i] += weight

        density[i] *= norm_factor / N

    return density


def _kde_cutoff_python(D: int) -> float:
    """Fallback implementation of kde_cutoff."""
    return 9.0 * (np.sqrt(D) + 1.0) ** 2


def _kde_bootstrap_error_python(
    X: NDArray[np.float64], n_bootstrap: int, bandwidth: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fallback implementation of kde_bootstrap_error."""
    N, D = X.shape
    grid = X.copy()

    boot_kdes = np.zeros((n_bootstrap, N), dtype=np.float64)

    for b in range(n_bootstrap):
        boot_sample = X[np.random.choice(N, N, replace=True)]
        boot_kdes[b] = _compute_kde_python(boot_sample, grid, bandwidth)

    mean_kde = np.mean(boot_kdes, axis=0)
    std_kde = np.std(boot_kdes, axis=0)

    return mean_kde, std_kde


def _kde_output_python(
    density: NDArray[np.float64], std_kde: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fallback implementation of kde_output."""
    prb = density
    aer = std_kde
    rer = std_kde / (density + 1e-8)

    return prb, aer, rer


# Public API
def gauss_prepare(
    X: NDArray[np.float64],
) -> Union[
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[
        NDArray[np.float64],  # mean
        NDArray[np.float64],  # cov
        NDArray[np.float64],  # inv_cov
        NDArray[np.float64],  # eigvals
        NDArray[np.float64],  # Hi
        NDArray[np.float64],  # Hiinv
    ],
]:
    """
    Computes parameters for Gaussian KDE.

    Parameters:
    - X: (N x D) Data points.

    Returns:
    - If using Cython implementation:
        mean, cov, inv_cov, eigvals, Hi, Hiinv
    - If using Python fallback:
        mean, cov
    """
    if _using_cython:
        return _gauss_prepare_cython(X)
    else:
        return _gauss_prepare_python(X)


def compute_kde(
    X: NDArray[np.float64], grid: NDArray[np.float64], bandwidth: Optional[float] = None
) -> NDArray[np.float64]:
    """
    Computes Kernel Density Estimation (KDE) on given grid points.

    Parameters:
    - X: (N, D) Data points.
    - grid: (G, D) Grid points where KDE is evaluated.
    - bandwidth: Bandwidth parameter for KDE.
                 Required for Python fallback implementation.
                 Ignored in Cython implementation which uses adaptive bandwidth.

    Returns:
    - density: (G,) KDE density values at each grid point.
    """
    if _using_cython:
        return _compute_kde_cython(X, grid)
    else:
        if bandwidth is None:
            # Default bandwidth if not provided (Scott's rule)
            N, D = X.shape
            bandwidth = N ** (-1.0 / (D + 4))
        return _compute_kde_python(X, grid, bandwidth)


def kde_cutoff(D: int) -> float:
    """
    Computes KDE cutoff (`kdecut2`) for the given dimensionality.

    Parameters:
    - D: Dimensionality of the data.

    Returns:
    - kdecut2: KDE squared cutoff.
    """
    if _using_cython:
        return _kde_cutoff(D)
    else:
        return _kde_cutoff_python(D)


def kde_bootstrap_error(
    X: NDArray[np.float64], n_bootstrap: int, bandwidth: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates KDE statistical error using bootstrap resampling.

    Parameters:
    - X: (N, D) Data points.
    - n_bootstrap: Number of bootstrap runs.
    - bandwidth: Bandwidth parameter. Required for Python fallback.

    Returns:
    - mean_kde: Mean KDE values.
    - std_kde: Standard deviation of KDE estimates.
    """
    if _using_cython:
        if bandwidth is None:
            # Default bandwidth if not provided (Scott's rule)
            N, D = X.shape
            bandwidth = N ** (-1.0 / (D + 4))
        return _kde_bootstrap_error(X, n_bootstrap, bandwidth)
    else:
        if bandwidth is None:
            # Default bandwidth if not provided (Scott's rule)
            N, D = X.shape
            bandwidth = N ** (-1.0 / (D + 4))
        return _kde_bootstrap_error_python(X, n_bootstrap, bandwidth)


def kde_output(
    density: NDArray[np.float64], std_kde: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    if _using_cython:
        return _kde_output(density, std_kde)
    else:
        return _kde_output_python(density, std_kde)
