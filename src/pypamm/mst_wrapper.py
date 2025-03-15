"""
Wrapper functions for the mst module.
"""

from typing import Union, List, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pypamm.mst import build_mst as _build_mst

def build_mst(X: ArrayLike, metric: str = "euclidean") -> NDArray[np.float64]:
    """
    Builds the Minimum Spanning Tree (MST) for the dataset using Kruskal's Algorithm.

    Parameters:
    - X: Data matrix (N x D)
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev", "cosine")

    Returns:
    - mst_edges: Array of MST edges [(i, j, distance), ...]
    
    Raises:
    - ValueError: If X is empty, has fewer than 2 points, or if an invalid metric is provided
    - TypeError: If X cannot be converted to a float array
    """
    # Check if X is empty
    if hasattr(X, 'size') and X.size == 0:
        raise ValueError("Input array X is empty")
    
    # Try to convert X to a numpy array
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert input to float array: {str(e)}")
    
    # Check if X has at least 2 points
    if X.shape[0] < 2:
        raise ValueError("MST requires at least 2 data points")
    
    # Check if metric is valid
    valid_metrics = ["euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski"]
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Valid options are: {', '.join(valid_metrics)}")
    
    # Call the Cython implementation
    return _build_mst(X, metric) 