#!/usr/bin/env python
"""
Minimum Spanning Tree (MST) Example

This example demonstrates how to use the build_mst function from pypamm
to construct a minimum spanning tree from a dataset and visualize it.

MSTs are useful for:
- Finding the most efficient way to connect all points
- Identifying clusters and their hierarchical structure
- Dimensionality reduction
- Feature extraction
- Outlier detection
"""

import numpy as np
import matplotlib.pyplot as plt
from pypamm import build_mst, select_grid_points

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with multiple clusters
def generate_clustered_data(n_samples=300, n_clusters=3):
    """Generate synthetic clustered data."""
    centers = np.random.rand(n_clusters, 2) * 10
    data = []
    
    # Generate points around each center
    points_per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        cluster_points = np.random.randn(points_per_cluster, 2) * 0.5 + centers[i]
        data.append(cluster_points)
    
    # Combine all clusters
    X = np.vstack(data)
    return X

# Generate data
X = generate_clustered_data(n_samples=300, n_clusters=3)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Create a figure for visualization
plt.figure(figsize=(15, 10))

# Plot the original data
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Original Data")
plt.xlabel("X")
plt.ylabel("Y")

# Build and visualize MST with different distance metrics
metrics = ["euclidean", "manhattan", "chebyshev"]

for i, metric in enumerate(metrics):
    # Build the MST
    mst_edges = build_mst(X, metric=metric)
    
    # Plot the results
    plt.subplot(2, 2, i+2)
    
    # Plot the points
    plt.scatter(X[:, 0], X[:, 1], s=30)
    
    # Plot the MST edges
    for edge in mst_edges:
        u, v = int(edge[0]), int(edge[1])
        plt.plot([X[u, 0], X[v, 0]], [X[u, 1], X[v, 1]], 'k-', alpha=0.5)
    
    plt.title(f"MST with {metric.capitalize()} Distance")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Print some statistics
    total_weight = sum(edge[2] for edge in mst_edges)
    avg_edge_weight = total_weight / len(mst_edges)
    print(f"\nMST with {metric} distance:")
    print(f"  - Number of edges: {len(mst_edges)}")
    print(f"  - Total weight: {total_weight:.2f}")
    print(f"  - Average edge weight: {avg_edge_weight:.2f}")

plt.tight_layout()
plt.savefig("mst_metrics_example.png")
print("Figure saved as 'mst_metrics_example.png'")

# Example: MST on grid points for large datasets
plt.figure(figsize=(15, 5))

# Generate a larger dataset
X_large = generate_clustered_data(n_samples=1000, n_clusters=5)

# Plot original data
plt.subplot(1, 3, 1)
plt.scatter(X_large[:, 0], X_large[:, 1], s=10, alpha=0.5)
plt.title(f"Original Data\n({X_large.shape[0]} points)")
plt.xlabel("X")
plt.ylabel("Y")

# Select grid points
grid_indices, grid_points = select_grid_points(X_large, ngrid=20)

# Plot grid points
plt.subplot(1, 3, 2)
plt.scatter(X_large[:, 0], X_large[:, 1], s=10, alpha=0.2, color='gray')
plt.scatter(grid_points[:, 0], grid_points[:, 1], s=30, color='red')
plt.title(f"Grid Points\n({grid_points.shape[0]} points)")
plt.xlabel("X")
plt.ylabel("Y")

# Build MST on grid points
mst_edges = build_mst(grid_points)

# Plot MST on grid points
plt.subplot(1, 3, 3)
plt.scatter(grid_points[:, 0], grid_points[:, 1], s=30)

for edge in mst_edges:
    u, v = int(edge[0]), int(edge[1])
    plt.plot([grid_points[u, 0], grid_points[v, 0]], 
             [grid_points[u, 1], grid_points[v, 1]], 'k-', alpha=0.7)

plt.title(f"MST on Grid Points\n({len(mst_edges)} edges)")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.savefig("mst_grid_example.png")
print("Figure saved as 'mst_grid_example.png'")

# Print explanation of MST and its applications
print("\nExplanation of Minimum Spanning Tree (MST):")
print("------------------------------------------")
print("A Minimum Spanning Tree (MST) is a subset of the edges of a connected,")
print("edge-weighted graph that connects all vertices together without cycles")
print("while minimizing the total edge weight.")

print("\nKey Properties of MST:")
print("- Contains exactly N-1 edges for N points")
print("- Has no cycles (it's a tree)")
print("- Connects all points (it's spanning)")
print("- Has minimum total edge weight among all spanning trees")

print("\nApplications of MST:")
print("1. Clustering:")
print("   - By removing the longest edges, the MST can be split into clusters")
print("   - Useful for identifying natural groupings in the data")

print("\n2. Dimensionality Reduction:")
print("   - MST preserves the local structure of the data")
print("   - Can be used as a preprocessing step for manifold learning")

print("\n3. Outlier Detection:")
print("   - Points connected by unusually long edges may be outliers")
print("   - The distribution of edge weights can reveal anomalies")

print("\n4. Feature Extraction:")
print("   - Properties of the MST (e.g., average edge weight, degree distribution)")
print("   - Can be used as features for machine learning")

print("\n5. Efficient Data Representation:")
print("   - MST provides a sparse representation of the data")
print("   - Useful for large datasets where full pairwise distances are expensive")

print("\nDistance Metrics and Their Effects:")
print("- Euclidean: Standard straight-line distance, good for most applications")
print("- Manhattan: Sum of absolute differences, sensitive to axis alignment")
print("- Chebyshev: Maximum difference along any dimension, creates more direct paths")

print("\nExample completed successfully!") 