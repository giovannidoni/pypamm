"""
Unit tests for the clustering utils module.
"""

import numpy as np
import pytest
from pypamm import merge_clusters

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
    
    # Print debug information
    print("Original labels:", cluster_labels)
    print("New labels:", new_labels)
    print("Point at index 3 (should be merged):", X[3])
    print("Distance from point 3 to cluster 0 center:", np.linalg.norm(X[3] - np.mean(X[cluster_labels == 0], axis=0)))
    print("Distance from point 3 to cluster 2 center:", np.linalg.norm(X[3] - np.mean(X[cluster_labels == 2], axis=0)))
    
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
    
    # Print debug information
    print("Original labels:", cluster_labels)
    print("New labels:", new_labels)
    print("Point at index 3 (should be merged):", X[3])
    print("Distance from point 3 to cluster 0 center:", np.linalg.norm(X[3] - np.mean(X[cluster_labels == 0], axis=0)))
    print("Distance from point 3 to cluster 2 center:", np.linalg.norm(X[3] - np.mean(X[cluster_labels == 2], axis=0)))
    
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

def test_merge_clusters_input_validation():
    """Test input validation of merge_clusters."""
    # Valid inputs
    X = np.array([[0, 0], [1, 1]], dtype=np.float64)
    cluster_labels = np.array([0, 1], dtype=np.int32)
    probabilities = np.array([0.5, 0.5], dtype=np.float64)
    covariance_matrices = np.array([np.eye(2), np.eye(2)], dtype=np.float64)
    
    # Test with invalid X dimension
    with pytest.raises(ValueError, match="X must be a 2D array"):
        merge_clusters(X[0], cluster_labels, probabilities, covariance_matrices)
    
    # Test with invalid cluster_labels dimension
    with pytest.raises(ValueError, match="cluster_labels must be a 1D array"):
        merge_clusters(X, np.array([[0], [1]]), probabilities, covariance_matrices)
    
    # Test with invalid probabilities dimension
    with pytest.raises(ValueError, match="probabilities must be a 1D array"):
        merge_clusters(X, cluster_labels, np.array([[0.5], [0.5]]), covariance_matrices)
    
    # Test with invalid covariance_matrices dimension
    with pytest.raises(ValueError, match="covariance_matrices must be a 3D array"):
        merge_clusters(X, cluster_labels, probabilities, np.eye(2))
    
    # Test with mismatched lengths
    with pytest.raises(ValueError, match="Length of cluster_labels must match the number of samples in X"):
        merge_clusters(X, np.array([0, 1, 2]), probabilities, covariance_matrices)
    
    # Test with mismatched probabilities length
    with pytest.raises(ValueError, match="Length of probabilities must match the number of samples in X"):
        merge_clusters(X, cluster_labels, np.array([0.5, 0.5, 0.5]), covariance_matrices)
    
    # Test with insufficient covariance matrices
    with pytest.raises(ValueError, match="Number of covariance matrices must be at least the number of clusters"):
        merge_clusters(X, np.array([0, 2]), probabilities, covariance_matrices)
    
    # Test with invalid threshold
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        merge_clusters(X, cluster_labels, probabilities, covariance_matrices, threshold=1.5)
    
    print("✅ test_merge_clusters_input_validation passed!")

if __name__ == "__main__":
    test_merge_clusters_basic()
    test_merge_clusters_with_specific_target()
    test_merge_clusters_no_merging()
    test_merge_clusters_input_validation()
    print("All tests passed!") 