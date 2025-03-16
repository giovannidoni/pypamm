# test_mst.py
import numpy as np

from pypamm.mst import build_mst


def test_build_mst():
    """
    Unit test for Minimum Spanning Tree (MST) construction.
    """
    # Generate a small synthetic dataset (5 points in 2D)
    X = np.array(
        [
            [0.0, 0.0],  # Point 0 at origin
            [1.0, 0.0],  # Point 1 at (1,0)
            [0.0, 1.0],  # Point 2 at (0,1)
            [1.0, 1.0],  # Point 3 at (1,1)
            [0.5, 0.5],  # Point 4 at (0.5,0.5) - center
        ],
        dtype=np.float64,
    )

    # Test with Euclidean distance
    mst_edges = build_mst(X, metric="euclidean")

    # Check if the MST has the correct number of edges (N-1)
    assert len(mst_edges) == X.shape[0] - 1, "MST should have N-1 edges"

    # Check if the MST edges are valid (contain valid vertex indices)
    for edge in mst_edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        assert 0 <= i < X.shape[0], f"Invalid vertex index {i}"
        assert 0 <= j < X.shape[0], f"Invalid vertex index {j}"
        assert dist > 0, "Distance should be positive"

    # Test with different distance metrics
    for metric in ["manhattan", "chebyshev", "cosine"]:
        mst_edges_alt = build_mst(X, metric=metric)
        assert len(mst_edges_alt) == X.shape[0] - 1, f"MST with {metric} metric should have N-1 edges"

    # Test with a larger dataset
    np.random.seed(42)  # For reproducibility
    X_large = np.random.rand(20, 2)
    mst_edges_large = build_mst(X_large, metric="euclidean")
    assert len(mst_edges_large) == X_large.shape[0] - 1, "MST for larger dataset should have N-1 edges"

    # Test MST properties
    def is_connected(edges, n):
        """Check if the graph is connected using Union-Find."""
        parent = list(range(n))

        def find(x):
            """Find the root of the set containing element x with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Merge the sets containing elements x and y."""
            parent[find(x)] = find(y)

        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            union(i, j)

        # Check if all vertices are in the same set
        root = find(0)
        return all(find(i) == root for i in range(n))

    # Convert mst_edges to a list of tuples (i, j)
    edge_list = [(int(edge[0]), int(edge[1])) for edge in mst_edges]

    # Check if the MST is connected
    assert is_connected(edge_list, X.shape[0]), "MST should be connected"

    # Check if the MST has no cycles (N-1 edges in a connected graph with N vertices implies no cycles)
    assert len(edge_list) == X.shape[0] - 1, "MST should have no cycles"
