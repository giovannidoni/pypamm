#!/usr/bin/env python
"""
Neighbor Graph Example

This example demonstrates how to use the neighbor graph functions from pypamm
to build different types of neighborhood graphs for a dataset.

Neighbor graphs are useful for:
- Capturing the structure and connectivity of data
- Preprocessing for clustering algorithms
- Dimensionality reduction
- Manifold learning
- Feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from pypamm import build_neighbor_graph, build_knn_graph

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset: 200 points in 2D with clusters
def generate_clustered_data(n_samples=200, n_clusters=3):
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
X = generate_clustered_data(n_samples=200, n_clusters=3)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Define different graph types to try
graph_types = ["knn", "gabriel", "relative_neighborhood", "delaunay"]

# Create a figure for visualization
plt.figure(figsize=(15, 10))

# Plot each graph type
for i, graph_type in enumerate(graph_types):
    plt.subplot(2, 2, i+1)
    
    # Build the graph
    if graph_type == "knn":
        # For KNN, use the specialized function
        k = 5
        graph = build_knn_graph(X, k=k)
        title = f"K-Nearest Neighbors (k={k})"
    else:
        # For other graph types, use the general function
        graph = build_neighbor_graph(X, graph_type=graph_type)
        title = f"{graph_type.replace('_', ' ').title()} Graph"
    
    # Plot the points
    plt.scatter(X[:, 0], X[:, 1], s=30)
    
    # Plot the edges
    rows, cols = graph.nonzero()
    for r, c in zip(rows, cols):
        plt.plot([X[r, 0], X[c, 0]], [X[r, 1], X[c, 1]], 'k-', alpha=0.2)
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Print some statistics
    n_edges = graph.nnz // 2  # Divide by 2 because the graph is symmetric
    avg_degree = graph.nnz / X.shape[0]
    print(f"{title}:")
    print(f"  - Number of edges: {n_edges}")
    print(f"  - Average node degree: {avg_degree:.2f}")
    print(f"  - Graph density: {graph.nnz / (X.shape[0] * X.shape[0]):.4f}")

plt.tight_layout()
plt.savefig("neighbor_graph_example.png")
print("Figure saved as 'neighbor_graph_example.png'")

# Print explanation of the different graph types
print("\nExplanation of Graph Types:")
print("---------------------------")
print("1. K-Nearest Neighbors (KNN):")
print("   - Connects each point to its k nearest neighbors")
print("   - Simple and efficient, but can create disconnected components")
print("   - Good for local structure, but may miss global structure")
print("   - Parameter k controls the density of connections")

print("\n2. Gabriel Graph:")
print("   - Two points are connected if no other point lies in the circle with")
print("     diameter defined by the two points")
print("   - Captures both local and global structure")
print("   - No parameters to tune")
print("   - Tends to create more connections than RNG but fewer than Delaunay")

print("\n3. Relative Neighborhood Graph (RNG):")
print("   - Two points are connected if no other point is closer to both of them")
print("   - Creates a sparser graph than Gabriel")
print("   - Preserves important structural features")
print("   - No parameters to tune")

print("\n4. Delaunay Triangulation:")
print("   - Creates a triangulation where no point is inside the circumcircle of any triangle")
print("   - Creates the densest graph among these options")
print("   - Captures global structure well")
print("   - No parameters to tune")
print("   - Often used as a starting point for other graph algorithms")

print("\nChoosing the Right Graph Type:")
print("- KNN: When you need control over the number of connections")
print("- Gabriel/RNG: When you want a balance between sparsity and connectivity")
print("- Delaunay: When you need a comprehensive graph that captures all relationships")

# Example of using the graph for a simple application: connected components
print("\nExample Application: Finding Connected Components")
graph = build_neighbor_graph(X, graph_type="gabriel")
n_components, labels = sparse.csgraph.connected_components(graph, directed=False)
print(f"The Gabriel graph has {n_components} connected components")

# Count points in each component
for i in range(n_components):
    print(f"Component {i+1} has {np.sum(labels == i)} points")

print("\nExample completed successfully!") 