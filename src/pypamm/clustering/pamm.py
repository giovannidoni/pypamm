import numpy as np
from typing import Tuple, List, Optional, Union, Any
from numpy.typing import NDArray, ArrayLike
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from pypamm.grid_selection import select_grid_points
from pypamm.neighbor_graph import build_neighbor_graph
from pypamm.quick_shift import quick_shift_clustering
from pypamm.clustering.cluster_utils import compute_cluster_covariance, merge_clusters
from pypamm.clustering.adjacency import compute_adjacency, merge

class PAMMCluster:
    """
    PAMM clustering pipeline with optional bootstrapping and adjacency-based merging.
    """

    def __init__(
        self, 
        n_grid: int = 20, 
        k_neighbors: int = 5, 
        metric: str = "euclidean", 
        merge_threshold: float = 0.8, 
        bootstrap: bool = True, 
        n_bootstrap: int = 10, 
        n_jobs: int = 1
    ):
        """
        Initialize the PAMM clustering algorithm.
        
        Parameters
        ----------
        n_grid : int, default=20
            Number of grid points to use for density estimation.
        k_neighbors : int, default=5
            Number of neighbors for graph construction.
        metric : str, default="euclidean"
            Distance metric to use.
        merge_threshold : float, default=0.8
            Threshold for merging clusters.
        bootstrap : bool, default=True
            Whether to use bootstrapping for robust clustering.
        n_bootstrap : int, default=10
            Number of bootstrap iterations.
        n_jobs : int, default=1
            Number of parallel jobs for bootstrapping.
        """
        self.n_grid = n_grid
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.merge_threshold = merge_threshold
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs  # Parallel processing

    def _single_run(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
        """
        Run a single clustering pipeline iteration.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        cluster_labels : array, shape (n_samples,)
            Cluster assignments for each data point.
        prob : array, shape (n_samples,)
            Probability values for each data point.
        """
        N, D = X.shape
        idx_grid, grid_points = select_grid_points(X, self.n_grid, self.metric)
        neighbor_graph = build_neighbor_graph(grid_points, self.k_neighbors, metric=self.metric)
        prob = np.ones(N) / N  # Uniform probabilities
        cluster_labels, cluster_centers = quick_shift_clustering(X, prob, self.n_grid, self.metric)
        cluster_covariances = np.array([np.eye(D) for _ in range(len(cluster_centers))])
        return cluster_labels, prob

    def fit(self, X: ArrayLike, single_iteration: bool = False) -> NDArray[np.int32]:
        """
        Run the full PAMM clustering pipeline.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        single_iteration : bool, default=False
            If True, runs only one clustering iteration (no bootstrapping).
            
        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster assignments for each data point.
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape

        # Step 1: Run a single iteration if bootstrapping is disabled
        if single_iteration or not self.bootstrap:
            return self._single_run(X)[0]  # Return cluster labels from a single clustering iteration

        # Step 2: Bootstrap multiple clusterings
        boot_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(X[np.random.choice(N, N, replace=True)]) 
            for _ in range(self.n_bootstrap)
        )

        # Step 3: Compute adjacency matrix from bootstrapped results
        cluster_labels_list, probs_list = zip(*boot_results)
        adj_matrix, _ = compute_adjacency(probs_list[0], cluster_labels_list[0])  # Compute adjacency

        # Step 4: Merge clusters based on adjacency matrix
        new_labels = merge(adj_matrix, cluster_labels_list[0], self.merge_threshold, N)

        return new_labels