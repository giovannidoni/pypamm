"""
Optimized operations library for PyPAMM.
"""

from pypamm.lib._opx import (
    compute_localization,
    detmatrix,
    effdim,
    eigval,
    factorial,
    invmatrix,
    logdet,
    maxeigval,
    oracle,
    trmatrix,
    wcovariance,
)
from pypamm.lib.distance import (
    compute_pairwise_distances,
    py_calculate_distance,
)
from pypamm.lib.distance_wrapper import get_distance_function

# Define what should be imported with "from pypamm.lib import *"
__all__ = [
    "invmatrix",
    "trmatrix",
    "detmatrix",
    "logdet",
    "eigval",
    "maxeigval",
    "factorial",
    "effdim",
    "oracle",
    "wcovariance",
    "compute_localization",
    "get_distance_function",
    "py_calculate_distance",
    "compute_pairwise_distances",
]
