"""
Wrapper functions for the quick_shift module.
"""

from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pypamm.quick_shift import quick_shift_clustering as _quick_shift_clustering

def quick_shift(
    X: ArrayLike, 
    prob: Optional[ArrayLike] = None, 
    ngrid: int = 100, 
    metric: str = "euclidean", 
    lambda_qs: float = 1.0, 
    max_dist: float = np.inf
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Quick-Shift clustering algorithm based on density gradient ascent.
    
    Parameters:
    - X: (N x D) NumPy array (data points)
    - prob: (N,) NumPy array of probability estimates for each point. If None, uniform probabilities are used.
    - ngrid: Number of grid points
    - metric: Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski")
    - lambda_qs: Scaling factor for density-based traversal
    - max_dist: Maximum distance threshold for connecting points (default: infinity)

    Returns:
    - idxroot: Cluster assignment for each point
    - cluster_centers: Array of unique cluster centers
    """
    # Ensure X is a numpy array
    X = np.asarray(X, dtype=np.float64)
    
    # If prob is None, use uniform probabilities
    if prob is None:
        prob = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
    else:
        prob = np.asarray(prob, dtype=np.float64)
    
    # Call the Cython implementation
    return _quick_shift_clustering(X, prob, ngrid, metric, lambda_qs, max_dist) 