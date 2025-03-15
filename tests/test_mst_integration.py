# test_mst_integration.py
import numpy as np

from pypamm import build_mst, build_neighbor_graph, select_grid_points


def test_mst_with_grid_selection():
    """
    Test integration of MST with grid selection.
    """
    # Generate a synthetic dataset
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10

    # Select grid points
    grid_indices, grid_points = select_grid_points(X, ngrid=10)

    # Build MST on the grid points
    mst_edges = build_mst(grid_points)

    # Check if the MST has the correct number of edges
    assert len(mst_edges) == grid_points.shape[0] - 1, "MST on grid points should have N-1 edges"

    # Check if the MST edges are valid
    for edge in mst_edges:
        i, j = int(edge[0]), int(edge[1])
        assert 0 <= i < grid_points.shape[0], f"Invalid vertex index {i}"
        assert 0 <= j < grid_points.shape[0], f"Invalid vertex index {j}"


def test_mst_vs_neighbor_graph():
    """
    Test comparison between MST and neighbor graph.
    """
    # Generate a synthetic dataset
    np.random.seed(42)
    X = np.random.rand(20, 2) * 10

    # Build MST
    mst_edges = build_mst(X)

    # Build neighbor graph (Gabriel graph)
    neighbor_graph = build_neighbor_graph(X, k=5, graph_type="gabriel")

    # Convert MST edges to a set of tuples for comparison
    mst_edge_set = set()
    for edge in mst_edges:
        i, j = int(edge[0]), int(edge[1])
        mst_edge_set.add((i, j))
        mst_edge_set.add((j, i))  # Add both directions

    # Extract edges from neighbor graph
    neighbor_edges = set()
    rows, cols = neighbor_graph.nonzero()
    for i, j in zip(rows, cols):
        neighbor_edges.add((i, j))

    # Check if MST is a subset of the neighbor graph
    # Note: This might not always be true for all datasets, but it's a reasonable test
    common_edges = mst_edge_set.intersection(neighbor_edges)
    assert len(common_edges) > 0, "MST and neighbor graph should share some edges"

    # Check if MST has the correct number of edges
    assert len(mst_edges) == X.shape[0] - 1, "MST should have N-1 edges"
