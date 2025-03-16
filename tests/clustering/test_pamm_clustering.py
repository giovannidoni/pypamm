import numpy as np

from pypamm.clustering.pamm import PAMMCluster


def test_pamm_single_run():
    X = np.random.rand(100, 5)  # 100 points in 5D
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=False)
    labels = pamm.fit(X, single_iteration=True)
    assert len(labels) == len(X)
    assert len(np.unique(labels)) > 0  # At least one cluster


def test_pamm_bootstrap_clusters():
    X = np.random.rand(100, 5)
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=True, n_bootstrap=5)
    labels = pamm.fit(X)
    assert len(labels) == len(X)
    assert len(np.unique(labels)) > 0  # At least one cluster


# def test_pamm_cluster_merging():
#     X = np.random.rand(100, 5)
#     pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=True, n_bootstrap=5, merge_threshold=0.5)
#     labels = pamm.fit(X)

#     assert len(set(labels)) < 100  # Clusters should merge


def test_pamm_consistency():
    X = np.random.rand(100, 5)
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=False)
    labels1 = pamm.fit(X, single_iteration=True)

    # Run again with the same data
    pamm = PAMMCluster(n_grid=10, k_neighbors=5, bootstrap=False)
    labels2 = pamm.fit(X, single_iteration=True)

    # Should get the same number of clusters
    assert len(np.unique(labels1)) == len(np.unique(labels2))
