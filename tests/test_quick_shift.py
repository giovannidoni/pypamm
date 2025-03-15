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
    
    # Test different distance metrics (excluding Minkowski which requires special handling)
    for metric in ["euclidean", "manhattan", "chebyshev", "cosine"]:
        idxroot, cluster_centers = quick_shift_clustering(X, prob, ngrid=5, metric=metric, max_dist=np.inf)
        
        # Check if cluster assignments exist for all points
        assert len(idxroot) == X.shape[0], "Each point must have a cluster assignment."
        
        # Check if cluster centers are a subset of points
        assert all(center in idxroot for center in cluster_centers), "Cluster centers must be valid indices."
        
        # Check if the number of clusters is within reasonable bounds
        assert 1 <= len(cluster_centers) <= X.shape[0], "Number of clusters should be between 1 and N."
    
    # Skip Minkowski test since it requires special handling of the inv_cov parameter
    # which is not directly supported in the quick_shift_clustering function
    
    # Test with a distance threshold
    max_dist = 2.0  # Only connect points within 2.0 distance units
    idxroot, cluster_centers = quick_shift_clustering(X, prob, ngrid=5, metric="euclidean", max_dist=max_dist)
    
    # Check if the number of clusters is reasonable with the distance threshold
    assert len(cluster_centers) >= 1, "Should have at least one cluster with distance threshold."
    
    print("✅ test_quick_shift_clustering passed!")
    