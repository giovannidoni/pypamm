"""
Python wrapper for the KDE Cython module.
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

# Import the Cython implementation
try:
    from pypamm.density.kde import (
        gauss_prepare as _gauss_prepare,
        compute_kde as _compute_kde,
        kde_cutoff as _kde_cutoff,
        kde_bootstrap_error as _kde_bootstrap_error,
        kde_output as _kde_output
    )
except ImportError:
    # Fallback implementations for testing or when Cython module is not available
    def _gauss_prepare(X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fallback implementation of gauss_prepare."""
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        return mean, cov
    
    def _compute_kde(X: NDArray[np.float64], grid: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
        """Fallback implementation of compute_kde."""
        N, D = X.shape
        G = grid.shape[0]
        density = np.zeros(G, dtype=np.float64)
        norm_factor = (1 / (np.sqrt(2 * np.pi) * bandwidth)) ** D
        
        for i in range(G):
            for j in range(N):
                dist_sq = np.sum((grid[i] - X[j]) ** 2)
                weight = np.exp(-0.5 * dist_sq / (bandwidth ** 2))
                density[i] += weight
            
            density[i] *= norm_factor / N
        
        return density
    
    def _kde_cutoff(D: int) -> float:
        """Fallback implementation of kde_cutoff."""
        return 9.0 * (np.sqrt(D) + 1.0) ** 2
    
    def _kde_bootstrap_error(X: NDArray[np.float64], n_bootstrap: int, bandwidth: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fallback implementation of kde_bootstrap_error."""
        N, D = X.shape
        grid = X.copy()
        
        boot_kdes = np.zeros((n_bootstrap, N), dtype=np.float64)
        
        for b in range(n_bootstrap):
            boot_sample = X[np.random.choice(N, N, replace=True)]
            boot_kdes[b] = _compute_kde(boot_sample, grid, bandwidth)
        
        mean_kde = np.mean(boot_kdes, axis=0)
        std_kde = np.std(boot_kdes, axis=0)
        
        return mean_kde, std_kde
    
    def _kde_output(density: NDArray[np.float64], std_kde: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Fallback implementation of kde_output."""
        prb = density
        aer = std_kde
        rer = std_kde / (density + 1e-8)
        
        return prb, aer, rer

# Public API
def gauss_prepare(X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes mean and covariance for Gaussian KDE.
    
    Parameters:
    - X: (N x D) Data points.
    
    Returns:
    - mean: (D,) Mean vector.
    - cov: (D x D) Covariance matrix.
    """
    return _gauss_prepare(X)

def compute_kde(X: NDArray[np.float64], grid: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    """
    Computes Kernel Density Estimation (KDE) on given grid points.

    Parameters:
    - X: (N, D) Data points.
    - grid: (G, D) Grid points where KDE is evaluated.
    - bandwidth: Bandwidth parameter for KDE.

    Returns:
    - density: (G,) KDE density values at each grid point.
    """
    return _compute_kde(X, grid, bandwidth)

def kde_cutoff(D: int) -> float:
    """
    Computes KDE cutoff (`kdecut2`) for the given dimensionality.

    Parameters:
    - D: Dimensionality of the data.
    
    Returns:
    - kdecut2: KDE squared cutoff.
    """
    return _kde_cutoff(D)

def kde_bootstrap_error(X: NDArray[np.float64], n_bootstrap: int, bandwidth: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    return _kde_bootstrap_error(X, n_bootstrap, bandwidth)

def kde_output(density: NDArray[np.float64], std_kde: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    return _kde_output(density, std_kde) 