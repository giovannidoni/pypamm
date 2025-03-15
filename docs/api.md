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

### `quick_shift(X, prob=None, ngrid=50, lambda_qs=1.0, max_dist=np.inf, metric="euclidean")`

Performs Quick Shift clustering on a dataset.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `prob` (numpy.ndarray, optional): Probability estimates for each point
- `ngrid` (int): Number of grid points for density estimation
- `lambda_qs` (float): Scaling factor for density-based traversal
- `max_dist` (float): Maximum distance threshold for connecting points
- `metric` (str): Distance metric to use

**Returns:**
- `cluster_labels` (numpy.ndarray): Cluster assignment for each point
- `cluster_centers` (numpy.ndarray): Indices of cluster centers

### `quick_shift_clustering(X, prob, ngrid, metric="euclidean", lambda_qs=1.0, max_dist=np.inf)`

Low-level implementation of the Quick Shift clustering algorithm.

**Parameters:**
- `X` (numpy.ndarray): Data matrix (N x D)
- `prob` (numpy.ndarray): Probability estimates for each point
- `ngrid` (int): Number of grid points
- `metric` (str): Distance metric to use
- `lambda_qs` (float): Scaling factor for density-based traversal
- `max_dist` (float): Maximum distance threshold for connecting points

**Returns:**
- `idxroot` (numpy.ndarray): Cluster assignment for each point
- `cluster_centers` (numpy.ndarray): Array of unique cluster centers

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
