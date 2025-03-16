# PyPAMM API Reference

This document provides a reference for the PyPAMM API.

## Table of Contents

1. [Grid Selection](#grid-selection)
2. [Neighbor Graph](#neighbor-graph)
3. [Quick Shift Clustering](#quick-shift-clustering)
4. [Minimum Spanning Tree (MST)](#minimum-spanning-tree-mst)
5. [Distance Metrics](#distance-metrics)

## Grid Selection

### `select_grid_points(X, ngrid=10)`

Selects a subset of points from a dataset based on a grid.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `ngrid` (int): Number of grid points along each dimension

**Returns:**
- `grid_indices` (numpy.ndarray): Indices of the selected grid points in the original dataset
- `grid_points` (numpy.ndarray): Coordinates of the selected grid points

## Neighbor Graph

### `build_neighbor_graph(X, graph_type="gabriel", k=5, metric="euclidean")`

Builds a neighborhood graph for a dataset.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `graph_type` (str): Type of graph to build ("knn", "gabriel", "relative_neighborhood", "delaunay")
- `k` (int): Number of neighbors for KNN graph (only used if graph_type="knn")
- `metric` (str): Distance metric to use

**Returns:**
- `graph` (scipy.sparse.csr_matrix): Adjacency matrix of the graph

### `build_knn_graph(X, k=5, metric="euclidean")`

Builds a k-nearest neighbor graph for a dataset.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `k` (int): Number of neighbors
- `metric` (str): Distance metric to use

**Returns:**
- `graph` (scipy.sparse.csr_matrix): Adjacency matrix of the graph

## Quick Shift Clustering

### `quick_shift(X, prob=None, ngrid=100, metric="euclidean", lambda_qs=1.0, max_dist=np.inf, neighbor_graph=None)`

Quick-Shift clustering algorithm based on density gradient ascent.

This implementation can work with either pairwise distances or a pre-computed
neighbor graph, automatically choosing the most efficient approach.

**Parameters:**
- `X` (array-like, shape (n_samples, n_features)): Input data points.
- `prob` (array-like, shape (n_samples,), optional): Probability estimates for each point. If None, uniform probabilities are used.
- `ngrid` (int, default=100): Number of grid points (only used when neighbor_graph is None).
- `metric` (str, default="euclidean"): Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski"). Only used when neighbor_graph is None.
- `lambda_qs` (float, default=1.0): Scaling factor for density-based traversal.
- `max_dist` (float, default=np.inf): Maximum distance threshold for connecting points. Only used when neighbor_graph is None.
- `neighbor_graph` (scipy.sparse matrix, optional): Pre-computed neighbor graph. If provided, this will be used instead of computing distances between all points, which is more efficient for large datasets.

**Returns:**
- `labels` (ndarray of shape (n_samples,)): Cluster assignment for each point.

### `quick_shift_kde(X, bandwidth, ngrid=100, metric="euclidean", lambda_qs=1.0, max_dist=np.inf, neighbor_graph=None)`

KDE-enhanced Quick-Shift clustering algorithm.

This implementation computes probability densities using Kernel Density Estimation (KDE)
before applying the Quick-Shift algorithm.

**Parameters:**
- `X` (array-like, shape (n_samples, n_features)): Input data points.
- `bandwidth` (float): Bandwidth parameter for KDE.
- `ngrid` (int, default=100): Number of grid points.
- `metric` (str, default="euclidean"): Distance metric ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski").
- `lambda_qs` (float, default=1.0): Scaling factor for density-based traversal.
- `max_dist` (float, default=np.inf): Maximum distance threshold for connecting points.
- `neighbor_graph` (scipy.sparse matrix, optional): Pre-computed neighbor graph. If provided, this will be used instead of computing distances between all points.

**Returns:**
- `labels` (ndarray of shape (n_samples,)): Cluster assignment for each point.

### `quick_shift_clustering(X, prob, ngrid, neighbor_graph=None, metric="euclidean", lambda_qs=1.0, max_dist=np.inf)`

Low-level implementation of the Quick-Shift clustering algorithm with optional graph constraints.

**Parameters:**
- `X` (array-like, shape (n_samples, n_features)): Input data points.
- `prob` (array-like, shape (n_samples,)): Probability estimates for each point.
- `ngrid` (int): Number of grid points.
- `neighbor_graph` (scipy.sparse matrix, optional): Pre-computed neighbor graph (MST, k-NN, Gabriel Graph).
- `metric` (str, default="euclidean"): Distance metric.
- `lambda_qs` (float, default=1.0): Scaling factor for density-based traversal.
- `max_dist` (float, default=np.inf): Maximum distance threshold.

**Returns:**
- `idxroot` (ndarray of shape (n_samples,)): Cluster assignment for each point.
- `cluster_centers` (ndarray): Array of unique cluster centers.

## Minimum Spanning Tree (MST)

### `build_mst(X, metric="euclidean")`

Builds the Minimum Spanning Tree (MST) for a dataset.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `metric` (str): Distance metric to use

**Returns:**
- `mst_edges` (list): List of tuples (i, j, distance) representing the edges of the MST

## Distance Metrics

### `get_distance_function(metric="euclidean")`

Returns a function that computes the specified distance metric.

**Parameters:**
- `metric` (str): Distance metric to use ("euclidean", "manhattan", "chebyshev", "cosine", "mahalanobis", "minkowski")

**Returns:**
- `distance_function` (callable): Function that computes the specified distance metric

## Note

This API reference is a placeholder. For more detailed information about each function, please refer to the docstrings in the source code.
