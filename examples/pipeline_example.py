#!/usr/bin/env python
"""
PyPAMM Pipeline Example

This example demonstrates how to combine multiple PyPAMM algorithms
in a pipeline for data analysis and clustering.

The pipeline includes:
1. Grid selection for data reduction
2. Building a neighbor graph on the grid points
3. Constructing a minimum spanning tree
4. Using Quick Shift for clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pypamm import (
    select_grid_points,
    build_neighbor_graph,
    build_mst,
    quick_shift
)

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with multiple clusters
def generate_complex_data(n_samples=1000):
    """Generate synthetic data with complex structure."""
    # Cluster 1: Gaussian cluster
    cluster1 = np.random.randn(n_samples // 4, 2) * 0.5 + np.array([2, 2])
    
    # Cluster 2: Elongated cluster
    x2 = np.random.randn(n_samples // 4, 1) * 1.5
    y2 = np.random.randn(n_samples // 4, 1) * 0.3
    cluster2 = np.hstack([x2 + 6, y2 + 5])
    
    # Cluster 3: Ring-shaped cluster
    theta = np.random.uniform(0, 2*np.pi, n_samples // 4)
    r = np.random.normal(2, 0.2, n_samples // 4)
    x3 = r * np.cos(theta) + 8
    y3 = r * np.sin(theta) + 1
    cluster3 = np.vstack([x3, y3]).T
    
    # Cluster 4: Noisy background
    cluster4 = np.random.rand(n_samples // 4, 2) * 10
    
    # Combine all clusters
    X = np.vstack([cluster1, cluster2, cluster3, cluster4])
    
    return X

# Generate data
X = generate_complex_data(n_samples=1000)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Create a figure for visualization
plt.figure(figsize=(15, 12))

# Step 1: Plot the original data
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
plt.title("1. Original Data")
plt.xlabel("X")
plt.ylabel("Y")
print("\nStep 1: Original data")
print(f"  - Number of points: {X.shape[0]}")

# Step 2: Select grid points to reduce the dataset
ngrid = 30
grid_indices, grid_points = select_grid_points(X, ngrid=ngrid)
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.2, color='gray')
plt.scatter(grid_points[:, 0], grid_points[:, 1], s=20, color='red')
plt.title("2. Grid Selection")
plt.xlabel("X")
plt.ylabel("Y")
print("\nStep 2: Grid selection")
print(f"  - Grid size: {ngrid}")
print(f"  - Number of grid points: {grid_points.shape[0]}")

# Step 3: Build a neighbor graph on the grid points
graph = build_neighbor_graph(grid_points, graph_type="gabriel")
plt.subplot(2, 2, 3)
plt.scatter(grid_points[:, 0], grid_points[:, 1], s=20)
rows, cols = graph.nonzero()
for r, c in zip(rows, cols):
    if r < c:  # Only plot each edge once
        plt.plot([grid_points[r, 0], grid_points[c, 0]], 
                 [grid_points[r, 1], grid_points[c, 1]], 'k-', alpha=0.3)
plt.title("3. Gabriel Graph on Grid Points")
plt.xlabel("X")
plt.ylabel("Y")
print("\nStep 3: Build neighbor graph")
print(f"  - Graph type: Gabriel")
print(f"  - Number of edges: {graph.nnz // 2}")  # Divide by 2 because the graph is symmetric
print(f"  - Average node degree: {graph.nnz / grid_points.shape[0]:.2f}")

# Step 4: Build MST on the grid points
mst_edges = build_mst(grid_points)
mst_graph = np.zeros((grid_points.shape[0], grid_points.shape[0]))
for edge in mst_edges:
    u, v = int(edge[0]), int(edge[1])
    mst_graph[u, v] = 1
    mst_graph[v, u] = 1

# Step 5: Estimate density for Quick Shift
kde = gaussian_kde(grid_points.T)
prob = kde(grid_points.T)
prob = prob / np.max(prob)  # Normalize to [0, 1]

# Step 6: Run Quick Shift clustering
cluster_labels, cluster_centers = quick_shift(
    grid_points, 
    prob=prob, 
    ngrid=20, 
    lambda_qs=1.0, 
    max_dist=3.0
)

# Plot the clustering results
plt.subplot(2, 2, 4)
unique_labels = np.unique(cluster_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# Plot points colored by cluster
for i, label in enumerate(unique_labels):
    mask = cluster_labels == label
    plt.scatter(grid_points[mask, 0], grid_points[mask, 1], 
                color=colors[i], s=30, alpha=0.7)

# Plot MST edges
for edge in mst_edges:
    u, v = int(edge[0]), int(edge[1])
    plt.plot([grid_points[u, 0], grid_points[v, 0]], 
             [grid_points[u, 1], grid_points[v, 1]], 'k-', alpha=0.3)

# Plot cluster centers
center_points = grid_points[cluster_centers.astype(int)]
plt.scatter(center_points[:, 0], center_points[:, 1], 
            c='red', s=100, marker='*', label='Cluster Centers')

plt.title("4. Quick Shift Clustering with MST")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

print("\nStep 4-6: MST and Quick Shift clustering")
print(f"  - Number of MST edges: {len(mst_edges)}")
print(f"  - Number of clusters: {len(unique_labels)}")
print(f"  - Number of cluster centers: {len(cluster_centers)}")

plt.tight_layout()
plt.savefig("pipeline_example.png")
print("\nFigure saved as 'pipeline_example.png'")

# Print explanation of the pipeline
print("\nPipeline Explanation:")
print("-------------------")
print("This pipeline demonstrates how to combine multiple PyPAMM algorithms:")
print("\n1. Data Generation:")
print("   - Created a synthetic dataset with complex structure")
print("   - Multiple clusters with different shapes and noise")

print("\n2. Grid Selection:")
print("   - Reduced the dataset from {X.shape[0]} to {grid_points.shape[0]} points".format(X=X, grid_points=grid_points))
print("   - Preserved the overall structure while making computation more efficient")

print("\n3. Neighbor Graph:")
print("   - Built a Gabriel graph on the grid points")
print("   - Captured the connectivity structure of the data")
print("   - Useful for understanding relationships between points")

print("\n4. Minimum Spanning Tree:")
print("   - Built an MST on the grid points")
print("   - Provided a sparse representation of the data")
print("   - Connected all points with minimum total edge weight")

print("\n5. Density Estimation:")
print("   - Estimated the density of the grid points using KDE")
print("   - Higher density indicates regions with more points")

print("\n6. Quick Shift Clustering:")
print("   - Used the density information to find clusters")
print("   - Automatically determined the number of clusters")
print("   - Identified cluster centers at density modes")

print("\nAdvantages of this Pipeline:")
print("- Handles large datasets efficiently through grid selection")
print("- Captures complex data structures with appropriate graph types")
print("- Finds clusters of arbitrary shape using density-based methods")
print("- Provides a hierarchical view of the data through the MST")
print("- Requires minimal parameter tuning")

print("\nExample completed successfully!") 