# test_mst_edge_cases.py
import numpy as np
import pytest
from pypamm import build_mst

def test_mst_small_datasets():
    """
    Test MST construction on very small datasets.
    """
    # Test with a single point
    X_single = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        # MST requires at least 2 points
        build_mst(X_single)
    
    # Test with two points
    X_two = np.array([[0.0, 0.0], [1.0, 1.0]])
    mst_edges = build_mst(X_two)
    assert len(mst_edges) == 1, "MST for 2 points should have 1 edge"
    assert int(mst_edges[0, 0]) == 0 and int(mst_edges[0, 1]) == 1, "MST edge should connect the two points"
    
    # Test with three points in a line
    X_three = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    mst_edges = build_mst(X_three)
    assert len(mst_edges) == 2, "MST for 3 points should have 2 edges"

def test_mst_identical_points():
    """
    Test MST construction with identical points.
    """
    # Test with duplicate points
    X_dup = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],  # Duplicate
        [2.0, 2.0]
    ])
    
    # This should work but might have zero-weight edges
    mst_edges = build_mst(X_dup)
    assert len(mst_edges) == X_dup.shape[0] - 1, "MST should have N-1 edges even with duplicate points"
    
    # Check if any edges have zero weight
    zero_weight_edges = [edge for edge in mst_edges if edge[2] == 0]
    assert len(zero_weight_edges) > 0, "MST with duplicate points should have at least one zero-weight edge"

def test_mst_invalid_inputs():
    """
    Test MST construction with invalid inputs.
    """
    # Test with empty array
    X_empty = np.array([])
    with pytest.raises(ValueError):
        build_mst(X_empty)
    
    # Test with invalid metric
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        build_mst(X, metric="invalid_metric")
    
    # Test with non-numeric data
    X_str = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(TypeError):
        build_mst(X_str) 