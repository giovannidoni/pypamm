"""
Python wrapper for the KDE Cython module.
"""

# Import the Cython implementation directly
from pypamm.density.kde import (
    compute_kde,
    gauss_prepare,
    kde_bootstrap_error,
    kde_cutoff,
    kde_output,
)

# Export all functions
__all__ = [
    "compute_kde",
    "gauss_prepare",
    "kde_bootstrap_error",
    "kde_cutoff",
    "kde_output",
]
