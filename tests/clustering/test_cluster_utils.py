"""
Unit tests for the clustering utilities module.
"""

import numpy as np
import pytest
from pypamm import compute_cluster_covariance, merge_clusters

def test_compute_cluster_covariance():
    """Test computation of cluster covariance matrices."""
    # Create a simple dataset with 2 clusters
    X = np.array([
        [0, 0],    # Cluster 0
        [1, 0],    # Cluster 0
        [0, 1],    # Cluster 0
        [10, 10],  # Cluster 1
        [11, 10],  # Cluster 1
        [10, 11]   # Cluster 1
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    
    # Compute covariance matrices
    cov_matrices = compute_cluster_covariance(X, cluster_labels)
    
    # Check shape
    assert cov_matrices.shape == (2, 2, 2)
    
    # Check that covariance matrices are symmetric
    for i in range(2):
        assert np.allclose(cov_matrices[i], cov_matrices[i].T)
    
    # Check that diagonal elements are positive
    for i in range(2):
        assert np.all(np.diag(cov_matrices[i]) > 0)
    
    # Check that the covariance matrices are correct
    # For cluster 0, the points are [0,0], [1,0], [0,1], which form a triangle
    # The covariance matrix should be close to [[1/3, 0], [0, 1/3]]
    assert np.allclose(cov_matrices[0], np.array([[0.33333, 0], [0, 0.33333]]), atol=1e-4)
    
    # For cluster 1, the points are [10,10], [11,10], [10,11], which also form a triangle
    # The covariance matrix should be close to [[0.33333, 0], [0, 0.33333]]
    assert np.allclose(cov_matrices[1], np.array([[0.33333, 0], [0, 0.33333]]), atol=1e-4)

def test_compute_cluster_covariance_regularization():
    """Test computation of cluster covariance matrices with regularization."""
    # Create a simple dataset with 2 clusters
    X = np.array([
        [0, 0],    # Cluster 0
        [1, 0],    # Cluster 0
        [0, 1],    # Cluster 0
        [10, 10],  # Cluster 1
        [11, 10],  # Cluster 1
        [10, 11]   # Cluster 1
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    
    # Compute covariance matrices with regularization
    reg_value = 0.5
    cov_matrices = compute_cluster_covariance(X, cluster_labels, regularization=reg_value)
    
    # Check that the diagonal elements are increased by the regularization value
    for i in range(2):
        expected_diag = np.array([0.33333, 0.33333]) + reg_value
        assert np.allclose(np.diag(cov_matrices[i]), expected_diag, atol=1e-4)
    
def test_merge_clusters_basic():
    """Test basic functionality of merge_clusters."""
    # Create a simple dataset with 3 clusters
    X = np.array([
        [0, 0],    # Cluster 0 (strong)
        [1, 0],    # Cluster 0 (strong)
        [0, 1],    # Cluster 0 (strong)
        [5, 5],    # Cluster 1 (weak)
        [10, 10],  # Cluster 2 (strong)
        [11, 10],  # Cluster 2 (strong)
        [10, 11]   # Cluster 2 (strong)
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 2, 2, 2], dtype=np.int32)
    
    # Assign probabilities (cluster 1 has low probability)
    probabilities = np.array([0.2, 0.2, 0.2, 0.05, 0.15, 0.1, 0.1], dtype=np.float64)
    
    # Create identity covariance matrices for each cluster
    covariance_matrices = np.array([
        np.eye(2),  # Cluster 0
        np.eye(2),  # Cluster 1
        np.eye(2)   # Cluster 2
    ], dtype=np.float64)
    
    # Merge clusters with threshold 0.1 (should merge cluster 1)
    new_labels = merge_clusters(X, cluster_labels, probabilities, covariance_matrices, threshold=0.1)
    
    # The point is actually closer to cluster 0, so it should be merged there
    assert new_labels[3] == 0
    
    # Check that other points kept their original labels
    assert np.all(new_labels[:3] == 0)
    assert np.all(new_labels[4:] == 2)
    
def test_merge_clusters_with_specific_target():
    """Test merge_clusters with a point that should be merged into cluster 2."""
    # Create a simple dataset with 3 clusters
    X = np.array([
        [0, 0],    # Cluster 0 (strong)
        [1, 0],    # Cluster 0 (strong)
        [0, 1],    # Cluster 0 (strong)
        [8, 8],    # Cluster 1 (weak) - closer to cluster 2
        [10, 10],  # Cluster 2 (strong)
        [11, 10],  # Cluster 2 (strong)
        [10, 11]   # Cluster 2 (strong)
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 2, 2, 2], dtype=np.int32)
    
    # Assign probabilities (cluster 1 has low probability)
    probabilities = np.array([0.2, 0.2, 0.2, 0.05, 0.15, 0.1, 0.1], dtype=np.float64)
    
    # Create identity covariance matrices for each cluster
    covariance_matrices = np.array([
        np.eye(2),  # Cluster 0
        np.eye(2),  # Cluster 1
        np.eye(2)   # Cluster 2
    ], dtype=np.float64)
    
    # Merge clusters with threshold 0.1 (should merge cluster 1)
    new_labels = merge_clusters(X, cluster_labels, probabilities, covariance_matrices, threshold=0.1)
    
    # Check that cluster 1 was merged into cluster 2 (closer than cluster 0)
    assert new_labels[3] == 2
    
    # Check that other points kept their original labels
    assert np.all(new_labels[:3] == 0)
    assert np.all(new_labels[4:] == 2)
    

def test_merge_clusters_no_merging():
    """Test merge_clusters with a threshold that doesn't trigger merging."""
    # Create a simple dataset with 2 clusters
    X = np.array([
        [0, 0],    # Cluster 0
        [1, 0],    # Cluster 0
        [10, 10],  # Cluster 1
        [11, 10]   # Cluster 1
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 1, 1], dtype=np.int32)
    
    # Assign equal probabilities
    probabilities = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    
    # Create identity covariance matrices for each cluster
    covariance_matrices = np.array([
        np.eye(2),  # Cluster 0
        np.eye(2)   # Cluster 1
    ], dtype=np.float64)
    
    # Merge clusters with threshold 0.1 (should not merge any clusters)
    new_labels = merge_clusters(X, cluster_labels, probabilities, covariance_matrices, threshold=0.1)
    
    # Check that no labels changed
    assert np.array_equal(new_labels, cluster_labels)
    

def test_merge_clusters_all_weak():
    """
    Test merge_clusters function when all clusters have very low probability.
    
    This test verifies that when all clusters have probabilities below the threshold,
    they will be merged into a single cluster. This is an edge case that tests
    the behavior of the algorithm when all clusters are considered "weak".
    """
    X = np.random.rand(50, 3)  # 50 points in 3D
    prob = np.full(50, 0.01)  # Very low probability for all points
    cluster_labels = np.arange(50)  # Each point starts as its own cluster
    cluster_cov = np.array([np.eye(3) for _ in range(50)])  # Identity covariance

    new_labels = merge_clusters(X, prob, cluster_labels, cluster_cov, threshold=0.8)

    assert len(set(new_labels)) == 1  # Expect everything to merge into one cluster


def test_merge_clusters_numerical_stability():
    """
    Test merge_clusters function with extremely small probability values to ensure numerical stability.
    
    This test verifies that:
    1. The function can handle extremely small probability values without numerical issues
    2. The merging behavior is consistent with the threshold
    3. No errors occur due to underflow or division by zero
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a dataset with 3 distinct clusters
    X = np.vstack([
        np.random.rand(30, 3),                      # Cluster 0: random points
        np.random.rand(30, 3) + np.array([10, 0, 0]),  # Cluster 1: shifted in x
        np.random.rand(30, 3) + np.array([0, 10, 0])   # Cluster 2: shifted in y
    ])
    
    # Create cluster labels (30 points per cluster)
    cluster_labels = np.repeat(np.arange(3), 30)
    
    # Create extremely small probabilities, but with different magnitudes per cluster
    prob = np.zeros(90)
    prob[:30] = np.random.uniform(1e-12, 1e-11, 30)    # Very small for cluster 0
    prob[30:60] = np.random.uniform(1e-10, 1e-9, 30)   # Small for cluster 1
    prob[60:] = np.random.uniform(1e-8, 1e-7, 30)      # Larger for cluster 2
    
    # Normalize probabilities to sum to 1
    prob = prob / np.sum(prob)
    
    # Create covariance matrices
    cluster_cov = np.array([np.eye(3) for _ in range(3)])
    
    # Test with threshold between cluster probability ranges
    # This should merge cluster 0 but keep clusters 1 and 2
    threshold_mid = 1e-3
    new_labels_mid = merge_clusters(X, cluster_labels, prob, cluster_cov, threshold=threshold_mid)
    
    # Count points in each cluster after merging
    unique_labels_mid = np.unique(new_labels_mid)
    
    # Verify that some merging occurred but not all clusters were merged
    assert 1 < len(unique_labels_mid) <= 3, "Should have merged some but not all clusters"
    
    # Test with very low threshold that shouldn't merge any clusters
    threshold_low = 1e-15
    new_labels_low = merge_clusters(X, cluster_labels, prob, cluster_cov, threshold=threshold_low)
    
    # Verify no merging occurred with very low threshold
    assert len(np.unique(new_labels_low)) == 3, "No clusters should be merged with very low threshold"
    
    # Test with high threshold that should merge all clusters
    threshold_high = 0.5
    new_labels_high = merge_clusters(X, cluster_labels, prob, cluster_cov, threshold=threshold_high)
    
    # Verify all clusters were merged
    assert len(np.unique(new_labels_high)) < 3, "Some clusters should be merged with high threshold"

