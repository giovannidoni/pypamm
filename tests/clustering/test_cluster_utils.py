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
    
    print("✅ test_compute_cluster_covariance passed!")

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
    
    print("✅ test_compute_cluster_covariance_regularization passed!")

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
    
    print("✅ test_merge_clusters_basic passed!")

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
    
    print("✅ test_merge_clusters_with_specific_target passed!")

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
    
    print("✅ test_merge_clusters_no_merging passed!")

if __name__ == "__main__":
    test_compute_cluster_covariance()
    test_compute_cluster_covariance_regularization()
    test_merge_clusters_basic()
    test_merge_clusters_with_specific_target()
    test_merge_clusters_no_merging()
    print("All tests passed!") 