import numpy as np
import pytest
from pypamm.clustering.adjacency import merge, compute_adjacency

def test_merge_clusters_above_threshold():
    adj_matrix = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    cluster_labels = np.array([0, 1, 2])
    new_labels = merge(adj_matrix, cluster_labels, thresh=0.8, N=3)

    assert new_labels[0] == new_labels[1]  # Clusters 0 and 1 should merge
    assert new_labels[2] != new_labels[0]  # Cluster 2 remains separate

def test_merge_clusters_below_threshold():
    adj_matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    cluster_labels = np.array([0, 1, 2])
    new_labels = merge(adj_matrix, cluster_labels, thresh=0.8, N=3)

    assert np.array_equal(new_labels, cluster_labels)  # No clusters should merge

def test_merge_invalid_clusters():
    adj_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
    cluster_labels = np.array([0, -1])
    new_labels = merge(adj_matrix, cluster_labels, thresh=0.8, N=2)

    assert new_labels[1] == -1  # Invalid cluster ID should remain unchanged

def test_compute_adjacency_self_adjacency():
    prob = np.array([0.2, 0.3, 0.5])
    clus = np.array([0, 1, 2])
    adj, pks = compute_adjacency(prob, clus)

    assert np.all(np.diag(adj) == 1.0)  # Self-adjacency should be 1

def test_compute_adjacency_probability_sum():
    prob = np.array([0.2, 0.3, 0.5])
    clus = np.array([0, 0, 1])
    adj, pks = compute_adjacency(prob, clus)

    assert np.isclose(np.sum(pks), 1.0)  # Probability masses should sum to 1

def test_compute_adjacency_high_overlap():
    prob = np.array([0.5, 0.5, 0.1, 0.1])
    clus = np.array([0, 0, 1, 1])
    adj, pks = compute_adjacency(prob, clus)

    assert adj[0, 1] > 0.1  # High overlap should result in high adjacency