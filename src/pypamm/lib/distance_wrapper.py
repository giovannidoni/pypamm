"""
Python wrapper for the distance_metrics Cython module.
"""

from functools import partial

from pypamm.lib.distance import py_calculate_distance


def get_distance_function(metric: str = "euclidean", inv_cov=None, k: int = 2) -> float:
    """
    Deprecated function, use calculate_distance directly.
    This function is maintained for backward compatibility only.

    Parameters:
    - metric: Distance metric name
    - inv_cov: Inverse covariance matrix for Mahalanobis distance
    - k: Parameter for Minkowski distance (p value)

    Returns:
    - Calculated distance value
    """

    return partial(py_calculate_distance, metric=metric, k=k, inv_cov=inv_cov)
