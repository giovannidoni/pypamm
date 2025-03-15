import numpy as np
import pytest
from pypamm.clustering.pamm import PAMMCluster

def test_pamm_single_run():
    X = np.random.rand(100, 5)  # 100 points in 5D
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=False)
    labels, _ = pamm.fit(X, single_iteration=True)

    assert len(set(labels)) > 1  # Expect more than one cluster

def test_pamm_bootstrap_clusters():
    X = np.random.rand(100, 5)
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=True, n_bootstrap=5)
    labels = pamm.fit(X)

    assert len(set(labels)) > 1  # Expect multiple clusters

def test_pamm_cluster_merging():
    X = np.random.rand(100, 5)
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=True, n_bootstrap=5, merge_threshold=0.5)
    labels = pamm.fit(X)

    assert len(set(labels)) < 100  # Clusters should merge

def test_pamm_consistency():
    X = np.random.rand(100, 5)
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=False)
    labels1, _ = pamm.fit(X, single_iteration=True)
    labels2, _ = pamm.fit(X, single_iteration=True)

    assert np.array_equal(labels1, labels2)  # Clustering should be stable