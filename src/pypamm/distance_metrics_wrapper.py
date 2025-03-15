"""
Python wrapper for the distance_metrics Cython module.
"""
from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray

def get_distance_function(metric: str = "euclidean") -> Callable[[NDArray[np.float64], NDArray[np.float64], Any], float]:
    """
    Get a distance function for the specified metric.
    
    Parameters:
    - metric: Distance metric name
    
    Returns:
    - distance_function: A function that computes the distance between two points
    """
    # Import the Cython implementation
    from pypamm.distance_metrics import get_distance_function as _get_distance_function
    
    # Call the Cython implementation
    return _get_distance_function(metric) 