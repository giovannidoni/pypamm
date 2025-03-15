"""
Test script for the quick_shift module.
"""

import numpy as np
from pypamm import quick_shift

# Generate some random data
np.random.seed(42)
X = np.random.randn(100, 2) * 0.5  # 100 points in 2D space with smaller variance

# Add some distinct clusters
X[:30] += np.array([5, 5])  # Cluster 1
X[30:60] += np.array([-5, 5])  # Cluster 2
X[60:] += np.array([0, -5])  # Cluster 3

# Create density estimates (higher values for points in cluster centers)
density = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    # Simple density estimate based on distance to other points
    dists = np.sum((X - X[i])**2, axis=1)
    density[i] = np.sum(np.exp(-dists / 0.5))  # Smaller bandwidth for sharper density peaks

# Normalize density
density = density / np.sum(density)

# Calculate average distance between clusters to set max_dist
cluster1_center = np.mean(X[:30], axis=0)
cluster2_center = np.mean(X[30:60], axis=0)
cluster3_center = np.mean(X[60:], axis=0)
avg_cluster_dist = (np.linalg.norm(cluster1_center - cluster2_center) + 
                   np.linalg.norm(cluster2_center - cluster3_center) + 
                   np.linalg.norm(cluster1_center - cluster3_center)) / 3

# Set max_dist to be less than the average distance between clusters
max_dist = avg_cluster_dist * 0.5
print(f"Setting max_dist to {max_dist:.2f} (half of average cluster distance {avg_cluster_dist:.2f})")

# Run quick_shift clustering with custom density and parameters
idxroot, cluster_centers = quick_shift(
    X, 
    prob=density, 
    ngrid=100, 
    lambda_qs=5.0,
    max_dist=max_dist
)

print(f"Number of clusters found: {len(cluster_centers)}")
print(f"Cluster centers: {cluster_centers}")

# Count points in each cluster
unique_clusters, counts = np.unique(idxroot, return_counts=True)
cluster_sizes = {}
for cluster, count in zip(unique_clusters, counts):
    cluster_sizes[cluster] = count
    print(f"Cluster ID {cluster}: {count} points")

# Find the main clusters (those with more than 5 points)
main_clusters = [cluster for cluster, count in cluster_sizes.items() if count > 5]
print(f"\nMain clusters (>5 points): {main_clusters}")

# Print cluster statistics
print("\nCluster Statistics:")
print(f"Total points: {X.shape[0]}")
print(f"Total clusters: {len(unique_clusters)}")
print(f"Average cluster size: {np.mean(counts):.2f}")
print(f"Largest cluster size: {np.max(counts)}")
print(f"Smallest cluster size: {np.min(counts)}")

# Check if we have roughly the expected number of main clusters (3)
if len(main_clusters) >= 2 and len(main_clusters) <= 5:
    print("\nSUCCESS: Found a reasonable number of main clusters!")
else:
    print("\nWARNING: Number of main clusters doesn't match expectations.")
    print("You may need to adjust the clustering parameters.")

print("\nQuick-shift clustering completed successfully!") 