# test_quick_shift.py
import numpy as np
from pypamm.quick_shift import quick_shift_clustering

def test_quick_shift_clustering():
    """
    Unit test for Quick-Shift clustering algorithm.
    """
    # Generate a small synthetic dataset (5 points in 2D)
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.5],
        [5.0, 1.0],
        [4.0, 2.0]
    ])
    
    # Assign synthetic probability estimates (higher probability for central points)
    prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    
    # Test different distance metrics
    for metric in ["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"]:
        idxroot, cluster_centers = quick_shift_clustering(X, prob, ngrid=5, metric=metric)
        
        # Check if cluster assignments exist for all points
        assert len(idxroot) == X.shape[0], "Each point must have a cluster assignment."
        
        # Check if cluster centers are a subset of points
        assert all(center in idxroot for center in cluster_centers), "Cluster centers must be valid indices."
        
        # Check if the number of clusters is within reasonable bounds
        assert 1 <= len(cluster_centers) <= X.shape[0], "Number of clusters should be between 1 and N."
    
    print("✅ test_quick_shift_clustering passed!")