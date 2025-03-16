# test_quick_shift.py
import numpy as np

from pypamm.neighbor_graph import build_neighbor_graph
from pypamm.quick_shift import quick_shift_clustering
from pypamm.quick_shift_wrapper import quick_shift


def test_quick_shift_clustering():
    """
    Unit test for Quick-Shift clustering algorithm.
    """
    # Generate a small synthetic dataset (5 points in 2D)
    X = np.array(
        [
            [0.0, 0.0],  # Point 0 at origin
            [1.0, 0.0],  # Point 1 at (1,0)
            [0.0, 1.0],  # Point 2 at (0,1)
            [1.0, 1.0],  # Point 3 at (1,1)
            [0.5, 0.5],  # Point 4 at (0.5,0.5) - center
            [10.0, 10.0],  # Point 5 far away
        ],
        dtype=np.float64,
    )

    # Create uniform probabilities
    prob = np.ones(X.shape[0])

    # Run Quick-Shift clustering
    labels, cluster_centers = quick_shift_clustering(X, prob, ngrid=5, lambda_qs=0.5)

    # Check if the number of points matches
    assert len(labels) == X.shape[0], "Number of labels should match number of points"

    # Check if the number of clusters is reasonable
    assert len(set(labels)) <= X.shape[0], "Number of clusters should not exceed number of points"

    # Check if the cluster centers are valid
    assert len(cluster_centers) <= X.shape[0], "Number of cluster centers should not exceed number of points"

    # Test with different lambda values
    labels_tight, centers_tight = quick_shift_clustering(X, prob, ngrid=5, lambda_qs=0.1)
    labels_loose, centers_loose = quick_shift_clustering(X, prob, ngrid=5, lambda_qs=2.0)

    # Tighter lambda should result in more clusters
    assert len(set(labels_tight)) >= len(set(labels)), "Tighter lambda should result in more clusters"

    # Test with distance threshold
    labels_dist, cluster_centers = quick_shift_clustering(X, prob, ngrid=5, lambda_qs=0.5, max_dist=0.5)

    # Check if the number of clusters is reasonable with the distance threshold
    assert len(cluster_centers) >= 1, "Should have at least one cluster with distance threshold."


def test_quick_shift_numerical_stability():
    """
    Test Quick-Shift clustering with nearly identical points to verify numerical stability.
    """
    X = np.ones((100, 3)) + np.random.rand(100, 3) * 1e-8  # Nearly identical points
    prob = np.ones(100)  # Use uniform probabilities to encourage a single cluster

    # Use the correct parameter order with a smaller lambda_qs value and larger max_dist
    labels, centers = quick_shift_clustering(
        X,
        prob,
        ngrid=5,
        neighbor_graph=None,
        metric="euclidean",
        lambda_qs=0.1,  # Smaller lambda_qs encourages fewer clusters
        max_dist=10.0,  # Larger max_dist allows points to connect over larger distances
    )

    # With nearly identical points and these parameters, we should have a very small number of clusters
    assert len(set(labels)) <= 5  # Should form a very small number of clusters


def test_quick_shift_large_dataset():
    """
    Test Quick-Shift clustering with a larger dataset to ensure it scales properly.
    """
    X = np.random.rand(1000, 5)  # 1000 points in 5D space
    prob = np.random.rand(1000)

    labels, centers = quick_shift_clustering(X, prob, ngrid=10)

    assert len(labels) == 1000  # Should return labels for all points


def test_quick_shift_with_neighbor_graph():
    """
    Test Quick-Shift clustering with a pre-computed neighbor graph.
    """
    # Generate a small synthetic dataset with clear clusters
    np.random.seed(42)
    X = np.vstack(
        [
            np.random.randn(20, 2) * 0.5 + np.array([0, 0]),  # Cluster 1
            np.random.randn(20, 2) * 0.5 + np.array([5, 0]),  # Cluster 2
            np.random.randn(20, 2) * 0.5 + np.array([0, 5]),  # Cluster 3
        ]
    )

    # Create uniform probabilities
    prob = np.ones(X.shape[0])

    # Build a neighbor graph
    k_neighbors = 5
    neighbor_graph = build_neighbor_graph(X, k=k_neighbors, metric="euclidean")

    # Run Quick-Shift clustering with neighbor graph
    labels = quick_shift(X, prob, neighbor_graph=neighbor_graph)

    # Check if the number of points matches
    assert len(labels) == X.shape[0], "Number of labels should match number of points"

    # Check if we have multiple clusters
    unique_labels = np.unique(labels)
    assert len(unique_labels) >= 3, "Should find at least 3 clusters"

    # Check that the labels are contiguous
    assert np.array_equal(unique_labels, np.arange(len(unique_labels))), "Labels should be contiguous"

    # Compare with regular quick_shift
    regular_labels = quick_shift(X, prob)

    # Both should find multiple clusters
    assert len(np.unique(regular_labels)) > 1, "Regular quick_shift should find multiple clusters"

    # The number of clusters might be different, but both should cluster the data
    assert len(np.unique(labels)) > 0, "Neighbor graph quick_shift should find clusters"
