#!/usr/bin/env python
"""
Grid Selection Example

This example demonstrates how to use the select_grid_points function from pypamm
to select a subset of points from a dataset based on a grid.

Grid selection is useful for:
- Reducing the number of points in large datasets
- Creating a more uniform distribution of points
- Preprocessing data for clustering algorithms
- Speeding up computations by working with a representative subset
"""

import matplotlib.pyplot as plt
import numpy as np

from pypamm import select_grid_points

# Set random seed for reproducibility
np.random.seed(42)


# use data_generator.py to generate a synthetic dataset
# Generate a synthetic dataset: 1000 points in 2D with clusters
def generate_clustered_data(n_samples=1000, n_clusters=5):
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
X = generate_clustered_data(n_samples=1000, n_clusters=5)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Select grid points with different grid sizes
grid_sizes = [10, 20, 50]

plt.figure(figsize=(15, 5))

# Plot original data
plt.subplot(1, len(grid_sizes) + 1, 1)
plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
plt.title(f"Original Data\n({X.shape[0]} points)")
plt.xlabel("X")
plt.ylabel("Y")

# Try different grid sizes
for i, ngrid in enumerate(grid_sizes):
    # Select grid points
    grid_indices, grid_points = select_grid_points(X, ngrid=ngrid)

    # Plot the results
    plt.subplot(1, len(grid_sizes) + 1, i + 2)

    # Plot original data as background
    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.2, color="gray")

    # Plot selected grid points
    plt.scatter(grid_points[:, 0], grid_points[:, 1], s=20, color="red")

    plt.title(f"Grid Selection\n(ngrid={ngrid}, {grid_points.shape[0]} points)")
    plt.xlabel("X")
    plt.ylabel("Y")

    print(f"Grid size {ngrid}: Selected {grid_points.shape[0]} points")

plt.tight_layout()
plt.savefig("grid_selection_example.png")
print("Figure saved as 'grid_selection_example.png'")

# Print explanation of the results
print("\nExplanation:")
print("------------")
print("The grid selection algorithm divides the space into a grid and selects")
print("representative points from each occupied grid cell.")
print("This creates a more uniform distribution of points while preserving")
print("the overall structure of the data.")
print("\nAs the grid size (ngrid) increases:")
print("- More grid cells are created")
print("- More points are selected")
print("- The selected points better represent the fine structure of the data")
print("- But computational cost for subsequent algorithms increases")
print("\nChoosing the right grid size depends on your specific application:")
print("- Smaller grid for faster computation and coarse representation")
print("- Larger grid for better preservation of data structure")

# Try without visualization
print("\nRunning grid selection without visualization...")
grid_indices, grid_points = select_grid_points(X, ngrid=30)
print(f"Selected {grid_points.shape[0]} points with ngrid=30")
print(f"Grid indices shape: {grid_indices.shape}")
print(f"Grid points shape: {grid_points.shape}")

print("\nExample completed successfully!")
