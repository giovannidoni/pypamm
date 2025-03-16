# test_mst_wrapper.py
import numpy as np

from pypamm import build_mst


def test_mst_wrapper():
    """
    Unit test for the MST wrapper function.
    """
    # Generate a small synthetic dataset (5 points in 2D)
    X = np.array(
        [
            [0.0, 0.0],  # Point 0 at origin
            [1.0, 0.0],  # Point 1 at (1,0)
            [0.0, 1.0],  # Point 2 at (0,1)
            [1.0, 1.0],  # Point 3 at (1,1)
            [0.5, 0.5],  # Point 4 at (0.5,0.5) - center
        ]
    )

    # Test with default parameters (Euclidean distance)
    mst_edges = build_mst(X)

    # Check if the MST has the correct number of edges (N-1)
    assert len(mst_edges) == X.shape[0] - 1, "MST should have N-1 edges"

    # Test with non-numpy array input
    X_list = X.tolist()
    mst_edges_list = build_mst(X_list)
    assert len(mst_edges_list) == len(X_list) - 1, "MST should work with list input"

    # Test with different distance metrics
    for metric in ["manhattan", "chebyshev", "cosine"]:
        mst_edges_alt = build_mst(X, metric=metric)
        assert len(mst_edges_alt) == X.shape[0] - 1, f"MST with {metric} metric should have N-1 edges"

    # Test with a 3D dataset
    X_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mst_edges_3d = build_mst(X_3d)
    assert len(mst_edges_3d) == X_3d.shape[0] - 1, "MST should work with 3D data"
