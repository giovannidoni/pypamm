#!/usr/bin/env python
"""
Graph-based Quick Shift Clustering Example

This example demonstrates how to use the quick_shift function with a pre-computed
neighbor graph for more efficient clustering of large datasets.

Key advantages of graph-based Quick Shift:
- Significantly faster for large datasets
- Memory efficient (avoids computing all pairwise distances)
- Can use different graph types to constrain the clustering
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from pypamm import build_knn_graph, build_neighbor_graph, quick_shift

# Set random seed for reproducibility
np.random.seed(42)


# Generate a synthetic dataset with multiple clusters
def generate_multi_cluster_data(n_samples=1000):
    """Generate synthetic data with multiple clusters of different shapes."""
    # Cluster 1: Gaussian cluster
    cluster1 = np.random.randn(n_samples // 4, 2) * 0.5 + np.array([2, 2])

    # Cluster 2: Elongated cluster
    x2 = np.random.randn(n_samples // 4, 1) * 1.5
    y2 = np.random.randn(n_samples // 4, 1) * 0.3
    cluster2 = np.hstack([x2 + 6, y2 + 5])

    # Cluster 3: Ring-shaped cluster
    theta = np.random.uniform(0, 2 * np.pi, n_samples // 4)
    r = np.random.normal(2, 0.2, n_samples // 4)
    x3 = r * np.cos(theta) + 8
    y3 = r * np.sin(theta) + 1
    cluster3 = np.vstack([x3, y3]).T

    # Cluster 4: Scattered points
    cluster4 = np.random.rand(n_samples // 4, 2) * 3 + np.array([4, 8])

    # Combine all clusters
    X = np.vstack([cluster1, cluster2, cluster3, cluster4])

    return X


# Generate data
X = generate_multi_cluster_data(n_samples=2000)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Estimate density using KDE
kde = gaussian_kde(X.T)
prob = kde(X.T)
prob = prob / np.max(prob)  # Normalize to [0, 1]

# Create a figure for visualization
plt.figure(figsize=(15, 12))

# Plot the original data with density
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=prob, cmap="viridis", s=30, alpha=0.7)
plt.colorbar(label="Normalized Density")
plt.title("Original Data with Density Estimation")
plt.xlabel("X")
plt.ylabel("Y")

# Compare different methods
methods = [("Standard Quick Shift", None), ("KNN Graph (k=10)", "knn"), ("Gabriel Graph", "gabriel")]

for i, (method_name, graph_type) in enumerate(methods):
    # Build neighbor graph if needed
    if graph_type is not None:
        if graph_type == "knn":
            print(f"\nBuilding {method_name}...")
            start_time = time.time()
            neighbor_graph = build_knn_graph(X, k=10)
            graph_time = time.time() - start_time
            print(f"  - Graph construction time: {graph_time:.4f} seconds")
            print(f"  - Number of edges: {neighbor_graph.nnz}")
        else:
            print(f"\nBuilding {method_name}...")
            start_time = time.time()
            neighbor_graph = build_neighbor_graph(X, graph_type=graph_type)
            graph_time = time.time() - start_time
            print(f"  - Graph construction time: {graph_time:.4f} seconds")
            print(f"  - Number of edges: {neighbor_graph.nnz}")
    else:
        neighbor_graph = None
        graph_time = 0
        print(f"\nUsing {method_name}...")

    # Run Quick Shift clustering
    print(f"Running Quick Shift clustering with {method_name}...")
    start_time = time.time()
    cluster_labels = quick_shift(X, prob=prob, ngrid=50, lambda_qs=1.0, max_dist=3.0, neighbor_graph=neighbor_graph)
    clustering_time = time.time() - start_time
    print(f"  - Clustering time: {clustering_time:.4f} seconds")
    print(f"  - Total time: {graph_time + clustering_time:.4f} seconds")

    # Plot the results
    plt.subplot(2, 2, i + 2)

    # Plot points colored by cluster
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for j, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(X[mask, 0], X[mask, 1], color=colors[j % 10], s=30, alpha=0.7)

    plt.title(f"{method_name}\n({len(unique_labels)} clusters, {clustering_time:.2f}s)")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Print some statistics
    print(f"  - Number of clusters: {len(unique_labels)}")

    # Count points in each cluster
    cluster_sizes = []
    for j, label in enumerate(unique_labels):
        count = np.sum(cluster_labels == label)
        cluster_sizes.append(count)

    print(f"  - Average cluster size: {np.mean(cluster_sizes):.1f} points")
    print(f"  - Largest cluster: {np.max(cluster_sizes)} points")
    print(f"  - Smallest cluster: {np.min(cluster_sizes)} points")

plt.tight_layout()
plt.savefig("quick_shift_graph_example.png")
print("\nFigure saved as 'quick_shift_graph_example.png'")

# Print explanation of the Graph-based Quick Shift algorithm
print("\nExplanation of Graph-based Quick Shift Clustering:")
print("-----------------------------------------------")
print("Graph-based Quick Shift is an optimized version of the Quick Shift algorithm that:")
print("1. Uses a pre-computed neighbor graph to restrict the search space")
print("2. Only considers connections that exist in the graph")
print("3. Significantly reduces computation time for large datasets")

print("\nAdvantages of Graph-based Quick Shift:")
print("- Much faster for large datasets")
print("- Memory efficient (avoids computing all pairwise distances)")
print("- Can use different graph types to constrain the clustering:")
print("  - KNN Graph: Fast and simple, good for most cases")
print("  - Gabriel Graph: Preserves natural neighbors, good for non-uniform data")
print("  - Delaunay Graph: Comprehensive connections, good for complete coverage")
print("  - RNG Graph: Sparse but connected, good balance of efficiency and quality")

print("\nExample completed successfully!")
