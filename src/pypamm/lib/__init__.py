"""
Optimized operations library for PyPAMM.
"""

from pypamm.lib._opx import (
    detmatrix,
    effdim,
    eigval,
    factorial,
    invmatrix,
    logdet,
    mahalanobis,
    maxeigval,
    oracle,
    pammrij,
    trmatrix,
    variance,
)

# Define what should be imported with "from pypamm.lib import *"
__all__ = [
    "invmatrix",
    "trmatrix",
    "detmatrix",
    "logdet",
    "variance",
    "eigval",
    "maxeigval",
    "factorial",
    "pammrij",
    "mahalanobis",
    "effdim",
    "oracle",
]
