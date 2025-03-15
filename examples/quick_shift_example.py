#!/usr/bin/env python
"""
Quick Shift Clustering Example

This example demonstrates how to use the quick_shift and quick_shift_clustering functions
from pypamm to perform mode-seeking clustering on a dataset.

Quick Shift is useful for:
- Finding clusters without specifying the number of clusters in advance
- Identifying clusters of arbitrary shape
- Handling noise and outliers
- Finding modes in the data distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pypamm import quick_shift, quick_shift_clustering

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with multiple clusters
def generate_multi_cluster_data(n_samples=500):
    """Generate synthetic data with multiple clusters of different shapes."""
    # Cluster 1: Gaussian cluster
    cluster1 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([2, 2])
    
    # Cluster 2: Elongated cluster
    x2 = np.random.randn(n_samples // 3, 1) * 1.5
    y2 = np.random.randn(n_samples // 3, 1) * 0.3
    cluster2 = np.hstack([x2 + 6, y2 + 5])
    
    # Cluster 3: Ring-shaped cluster
    theta = np.random.uniform(0, 2*np.pi, n_samples // 3)
    r = np.random.normal(2, 0.2, n_samples // 3)
    x3 = r * np.cos(theta) + 8
    y3 = r * np.sin(theta) + 1
    cluster3 = np.vstack([x3, y3]).T
    
    # Combine all clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    
    return X

# Generate data
X = generate_multi_cluster_data(n_samples=600)
print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

# Estimate density using KDE
kde = gaussian_kde(X.T)
prob = kde(X.T)
prob = prob / np.max(prob)  # Normalize to [0, 1]

# Create a figure for visualization
plt.figure(figsize=(15, 10))

# Plot the original data with density
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=prob, cmap='viridis', s=30)
plt.colorbar(label='Normalized Density')
plt.title("Original Data with Density Estimation")
plt.xlabel("X")
plt.ylabel("Y")

# Try different lambda values for Quick Shift
lambda_values = [0.5, 1.0, 2.0]
max_dist = 3.0  # Maximum distance threshold

for i, lambda_qs in enumerate(lambda_values):
    # Run Quick Shift clustering
    cluster_labels, cluster_centers = quick_shift(
        X, 
        prob=prob, 
        ngrid=50, 
        lambda_qs=lambda_qs, 
        max_dist=max_dist
    )
    
    # Plot the results
    plt.subplot(2, 2, i+2)
    
    # Plot points colored by cluster
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for j, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(X[mask, 0], X[mask, 1], color=colors[j], s=30, alpha=0.7, label=f"Cluster {j+1}")
    
    # Plot cluster centers
    center_points = X[cluster_centers.astype(int)]
    plt.scatter(center_points[:, 0], center_points[:, 1], c='red', s=100, marker='*', label='Cluster Centers')
    
    plt.title(f"Quick Shift Clustering\n(lambda={lambda_qs}, {len(unique_labels)} clusters)")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Print some statistics
    print(f"\nQuick Shift with lambda={lambda_qs}:")
    print(f"  - Number of clusters: {len(unique_labels)}")
    print(f"  - Number of cluster centers: {len(cluster_centers)}")
    
    # Count points in each cluster
    for j, label in enumerate(unique_labels):
        count = np.sum(cluster_labels == label)
        print(f"  - Cluster {j+1}: {count} points")

plt.tight_layout()
plt.savefig("quick_shift_example.png")
print("Figure saved as 'quick_shift_example.png'")

# Print explanation of the Quick Shift algorithm
print("\nExplanation of Quick Shift Clustering:")
print("------------------------------------")
print("Quick Shift is a mode-seeking clustering algorithm that:")
print("1. Estimates the density of the data points")
print("2. Shifts each point towards the nearest neighbor with higher density")
print("3. Forms clusters by connecting points that shift to the same mode")

print("\nKey Parameters:")
print("- lambda_qs: Controls the tradeoff between density and distance")
print("  - Smaller values prioritize density (more clusters)")
print("  - Larger values prioritize distance (fewer clusters)")
print("- max_dist: Maximum distance threshold for connecting points")
print("  - Limits the maximum shift distance")
print("  - Helps prevent connecting distant clusters")
print("- ngrid: Number of grid points for density estimation")
print("  - Higher values provide more accurate density estimation")

print("\nAdvantages of Quick Shift:")
print("- Automatically determines the number of clusters")
print("- Can find clusters of arbitrary shape")
print("- Robust to noise and outliers")
print("- Based on density estimation, which is intuitive")

print("\nLimitations:")
print("- Sensitive to parameter settings")
print("- Requires density estimation, which can be computationally expensive")
print("- May struggle with clusters of varying densities")

print("\nExample completed successfully!") 