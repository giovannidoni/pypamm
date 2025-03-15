# test_quick_shift.py
import numpy as np
from pypamm.quick_shift import quick_shift_clustering

def test_quick_shift_clustering():
    """
    Unit test for Quick-Shift clustering algorithm.
    """
    # Generate a small synthetic dataset (5 points in 2D)
    X = np.array([
        [0.0, 0.0],  # Point 0 at origin
        [1.0, 0.0],  # Point 1 at (1,0)
        [0.0, 1.0],  # Point 2 at (0,1)
        [1.0, 1.0],  # Point 3 at (1,1)
        [0.5, 0.5],  # Point 4 at (0.5,0.5) - center
        [10.0, 10.0],  # Point 5 far away
    ], dtype=np.float64)
    
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
    prob = np.random.rand(100)

    labels, centers = quick_shift_clustering(X, prob, ngrid=5)

    assert len(set(labels)) == 1  # Should form one cluster


def test_quick_shift_large_dataset():
    """
    Test Quick-Shift clustering with a larger dataset to ensure it scales properly.
    """
    X = np.random.rand(1000, 5)  # 1000 points in 5D space
    prob = np.random.rand(1000)
    
    labels, centers = quick_shift_clustering(X, prob, ngrid=10)
    
    assert len(labels) == 1000  # Should return labels for all points