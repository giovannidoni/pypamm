from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from pypamm.clustering.adjacency import compute_adjacency, merge
from pypamm.clustering.cluster_utils_wrapper import reindex_clusters
from pypamm.density.kde import compute_kde
from pypamm.distance_metrics import get_distance_function
from pypamm.grid_selection import select_grid_points
from pypamm.neighbor_graph import build_neighbor_graph
from pypamm.quick_shift_wrapper import quick_shift


class PAMMCluster:
    """
    PAMM clustering pipeline using Cython for optimized performance.
    """

    def __init__(
        self,
        n_grid: int = 20,
        k_neighbors: int = 5,
        metric: str = "euclidean",
        merge_threshold: float = 0.8,
        bootstrap: bool = True,
        n_bootstrap: int = 10,
        n_jobs: int = 1,
        use_neighbor_graph: bool = False,
        bandwidth: float = 1.0,
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
        use_neighbor_graph : bool, default=False
            Whether to use neighbor graph for clustering.
        bandwidth : float, default=1.0
            Bandwidth parameter for KDE.
        """
        self.n_grid = n_grid
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.merge_threshold = merge_threshold
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs  # Parallel processing
        self.use_neighbor_graph = use_neighbor_graph
        self.bandwidth = bandwidth

        # Properties that will be set during fitting
        self.cluster_centers_ = None
        self.cluster_labels_ = None
        self.probabilities_ = None
        self.kde_density_ = None
        self.adjacency_matrix_ = None
        self.n_clusters_ = None
        self.bootstrap_labels_ = None
        self.bootstrap_probabilities_ = None
        self.neighbor_graph_ = None

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
        _, grid_points = select_grid_points(X, self.n_grid, self.metric)

        # Compute KDE for density estimation
        prob = compute_kde(X, X, self.bandwidth, adaptive=False)
        self.kde_density_ = prob  # Store KDE values for later use

        # Use neighbor graph if requested
        if self.use_neighbor_graph:
            neighbor_graph = build_neighbor_graph(X, k=self.k_neighbors, metric=self.metric)
            self.neighbor_graph_ = neighbor_graph  # Store for later use
            cluster_labels = quick_shift(X, prob, neighbor_graph=neighbor_graph)
        else:
            cluster_labels = quick_shift(X, prob)

        # Compute cluster centers
        unique_labels = np.unique(cluster_labels)
        self.n_clusters_ = len(unique_labels)
        self.cluster_centers_ = np.array([np.mean(X[cluster_labels == i], axis=0) for i in unique_labels])

        return cluster_labels, prob

    def perform_bootstrapping(
        self, X: NDArray[np.float64]
    ) -> tuple[list[NDArray[np.int32]], list[NDArray[np.float64]]]:
        """
        Generate multiple clusterings through bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        cluster_labels_list : list of arrays
            List of cluster label arrays from bootstrap iterations.
        probabilities_list : list of arrays
            List of probability arrays from bootstrap iterations.
        """
        N = len(X)
        boot_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(X[np.random.choice(N, N, replace=True)]) for _ in range(self.n_bootstrap)
        )

        cluster_labels_list, probabilities_list = zip(*boot_results)

        # Save bootstrap results
        self.bootstrap_labels_ = list(cluster_labels_list)
        self.bootstrap_probabilities_ = list(probabilities_list)

        return list(cluster_labels_list), list(probabilities_list)

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
        N = len(X)

        if single_iteration or not self.bootstrap:
            self.cluster_labels_, self.probabilities_ = self._single_run(X)
        else:
            cluster_labels_list, probabilities_list = self.perform_bootstrapping(X)

            # Compute adjacency matrix
            adjacency_matrix, prob_mass = compute_adjacency(probabilities_list[0], cluster_labels_list[0])
            self.adjacency_matrix_ = adjacency_matrix

            # Merge clusters based on adjacency
            merged_labels = merge(adjacency_matrix, cluster_labels_list[0], self.merge_threshold, N)

            # Reindex cluster labels
            self.cluster_labels_ = reindex_clusters(merged_labels)
            self.probabilities_ = probabilities_list[0]  # Use probabilities from first bootstrap

            # Compute and store cluster centers
            unique_labels = np.unique(self.cluster_labels_)
            self.n_clusters_ = len(unique_labels)
            self.cluster_centers_ = np.array([np.mean(X[self.cluster_labels_ == i], axis=0) for i in unique_labels])

        return self.cluster_labels_

    def _predict(self, X_new: ArrayLike) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """
        Predicts cluster assignments and return KDE-based probability estimates.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New data points to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster assignments.
        probabilities : array, shape (n_samples,)
            KDE-based probability estimates.
        """
        if self.cluster_centers_ is None or self.kde_density_ is None:
            raise ValueError("Model has not been fitted yet.")

        X_new = np.asarray(X_new, dtype=np.float64)
        N_new, D = X_new.shape

        # Get unique cluster labels and corresponding KDE values
        unique_clusters = np.unique(self.cluster_labels_)
        cluster_kde_means = np.array([np.mean(self.kde_density_[self.cluster_labels_ == c]) for c in unique_clusters])

        # Get the appropriate distance function
        dist_func = get_distance_function(self.metric)

        # Assign new points based on KDE-weighted distances
        labels = np.zeros(N_new, dtype=np.int32)
        probabilities = np.zeros(N_new, dtype=np.float64)

        for i in range(N_new):
            min_dist = np.inf
            best_cluster = -1
            best_prob = 0.0

            for j, cluster_id in enumerate(unique_clusters):
                # Compute KDE ratio (use stored KDE, not recomputed)
                kde_ratio = self.kde_density_.mean() / (cluster_kde_means[j] + 1e-6)

                # Compute KDE-weighted distance
                weighted_dist = dist_func(X_new[i], self.cluster_centers_[j], self.bandwidth) * kde_ratio

                # Assign to closest cluster
                if weighted_dist < min_dist:
                    min_dist = weighted_dist
                    best_cluster = cluster_id
                    best_prob = np.exp(-weighted_dist)  # Convert distance to probability

            labels[i] = best_cluster
            probabilities[i] = best_prob

        return labels, probabilities

    def predict(self, X_new: ArrayLike) -> NDArray[np.int32]:
        """
        Predicts cluster assignments for new data using KDE-weighted distances.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster assignments.
        """
        return self._predict(X_new)[0]

    def predict_proba(self, X_new: ArrayLike) -> NDArray[np.float64]:
        """
        Predicts cluster assignments probabilities for new data using KDE-weighted distances.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        probabilities : array, shape (n_samples,)
            Cluster assignments probabilites.
        """
        return self._predict(X_new)[1]

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int32]:
        """
        Fit the model and predict cluster labels for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster and predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.cluster_labels_

    def get_cluster_centers(self) -> NDArray[np.float64]:
        """
        Get the cluster centers.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            Cluster centers.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.cluster_centers_

    def get_adjacency_matrix(self) -> NDArray[np.float64]:
        """
        Get the adjacency matrix between clusters.

        Returns
        -------
        adjacency_matrix : array, shape (n_clusters, n_clusters)
            Adjacency matrix between clusters.
        """
        if not hasattr(self, "adjacency_matrix_") or self.adjacency_matrix_ is None:
            raise ValueError("Model has not been fitted with bootstrapping.")
        return self.adjacency_matrix_
