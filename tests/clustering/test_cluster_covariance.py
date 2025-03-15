"""
Unit tests for the cluster_covariance module.
"""

import numpy as np
import pytest
from pypamm import compute_cluster_covariance

def test_compute_cluster_covariance_basic():
    """Test basic functionality of compute_cluster_covariance."""
    # Create a simple dataset with 2 clusters
    X = np.array([
        [0, 0],  # Cluster 0
        [1, 0],  # Cluster 0
        [0, 1],  # Cluster 0
        [10, 10],  # Cluster 1
        [11, 10],  # Cluster 1
        [10, 11]   # Cluster 1
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    
    # Compute covariance matrices
    cov_matrices = compute_cluster_covariance(X, cluster_labels)
    
    # Check shape
    assert cov_matrices.shape == (2, 2, 2)
    
    # Manually compute expected covariance matrices
    cluster0_points = X[0:3]
    cluster1_points = X[3:6]
    
    expected_cov0 = np.cov(cluster0_points.T)
    expected_cov1 = np.cov(cluster1_points.T)
    
    # Check values (allowing for small numerical differences)
    np.testing.assert_allclose(cov_matrices[0], expected_cov0, rtol=1e-5)
    np.testing.assert_allclose(cov_matrices[1], expected_cov1, rtol=1e-5)
    
    print("✅ test_compute_cluster_covariance_basic passed!")

def test_compute_cluster_covariance_regularization():
    """Test regularization parameter of compute_cluster_covariance."""
    # Create a simple dataset with 2 clusters
    X = np.array([
        [0, 0],  # Cluster 0
        [1, 0],  # Cluster 0
        [0, 1],  # Cluster 0
        [10, 10],  # Cluster 1
        [11, 10],  # Cluster 1
        [10, 11]   # Cluster 1
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    
    # Regularization value
    reg_value = 0.5
    
    # Compute covariance matrices with regularization
    cov_matrices_reg = compute_cluster_covariance(X, cluster_labels, regularization=reg_value)
    
    # Compute covariance matrices without regularization
    cov_matrices = compute_cluster_covariance(X, cluster_labels)
    
    # Check that diagonal elements are increased by reg_value
    for i in range(2):  # For each cluster
        diag_diff = np.diag(cov_matrices_reg[i]) - np.diag(cov_matrices[i])
        np.testing.assert_allclose(diag_diff, reg_value, rtol=1e-10)
    
    print("✅ test_compute_cluster_covariance_regularization passed!")

def test_compute_cluster_covariance_input_validation():
    """Test input validation of compute_cluster_covariance."""
    # Valid inputs
    X = np.array([[0, 0], [1, 1]], dtype=np.float64)
    cluster_labels = np.array([0, 1], dtype=np.int32)
    
    # Test with invalid X dimension
    with pytest.raises(ValueError, match="X must be a 2D array"):
        compute_cluster_covariance(X[0], cluster_labels)
    
    # Test with invalid cluster_labels dimension
    with pytest.raises(ValueError, match="cluster_labels must be a 1D array"):
        compute_cluster_covariance(X, np.array([[0], [1]]))
    
    # Test with mismatched lengths
    with pytest.raises(ValueError, match="Length of cluster_labels must match the number of samples in X"):
        compute_cluster_covariance(X, np.array([0, 1, 2]))
    
    # Test with invalid regularization type
    with pytest.raises(ValueError, match="regularization must be a number"):
        compute_cluster_covariance(X, cluster_labels, regularization="invalid")
    
    # Test with negative regularization
    with pytest.raises(ValueError, match="regularization must be non-negative"):
        compute_cluster_covariance(X, cluster_labels, regularization=-1.0)
    
    print("✅ test_compute_cluster_covariance_input_validation passed!")

def test_compute_cluster_covariance_empty_clusters():
    """Test compute_cluster_covariance with empty clusters."""
    # Create a dataset with a gap in cluster labels (cluster 1 is empty)
    X = np.array([
        [0, 0],  # Cluster 0
        [1, 0],  # Cluster 0
        [10, 10],  # Cluster 2
        [11, 10]   # Cluster 2
    ], dtype=np.float64)
    
    cluster_labels = np.array([0, 0, 2, 2], dtype=np.int32)
    
    # Compute covariance matrices
    cov_matrices = compute_cluster_covariance(X, cluster_labels)
    
    # Check shape (should have 3 clusters: 0, 1 (empty), and 2)
    assert cov_matrices.shape == (3, 2, 2)
    
    # Check that cluster 1's covariance matrix is all zeros
    np.testing.assert_allclose(cov_matrices[1], np.zeros((2, 2)), rtol=1e-10)
    
    print("✅ test_compute_cluster_covariance_empty_clusters passed!")

if __name__ == "__main__":
    test_compute_cluster_covariance_basic()
    test_compute_cluster_covariance_regularization()
    test_compute_cluster_covariance_input_validation()
    test_compute_cluster_covariance_empty_clusters()
    print("All tests passed!") 