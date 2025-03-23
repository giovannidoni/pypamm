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
    maxeigval,
    oracle,
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
    "effdim",
    "oracle",
]
