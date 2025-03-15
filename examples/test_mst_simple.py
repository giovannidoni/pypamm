#!/usr/bin/env python
"""
Simple test script for the MST module in pypamm.
This script demonstrates how to use the build_mst function to create a minimum spanning tree
from a set of points, without using matplotlib for visualization.
"""

import numpy as np
from pypamm import build_mst

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset: 20 points in 2D
X = np.random.rand(20, 2) * 10
print(f"Generated {X.shape[0]} points in {X.shape[1]}D space")

# Build the MST
edges = build_mst(X)
print(f"MST contains {len(edges)} edges")

# Print the first few edges
print("\nFirst 5 edges of the MST:")
for i, (u, v, w) in enumerate(edges[:5]):
    print(f"Edge {i+1}: Point {int(u)} to Point {int(v)} with weight {w:.4f}")

# Calculate the total weight of the MST
total_weight = sum(w for _, _, w in edges)
print(f"\nTotal MST weight: {total_weight:.4f}")

# Verify MST properties
# 1. Number of edges should be n-1 for n points
assert len(edges) == X.shape[0] - 1, f"MST should have {X.shape[0] - 1} edges, but has {len(edges)}"

# 2. All vertex indices should be valid
all_vertices = set()
for u, v, _ in edges:
    all_vertices.add(int(u))
    all_vertices.add(int(v))
assert len(all_vertices) == X.shape[0], f"MST should include all {X.shape[0]} vertices"

# 3. Check if the MST is connected (all vertices are reachable)
# We can use a simple Union-Find data structure for this
parent = list(range(X.shape[0]))

def find(x):
    x = int(x)  # Convert float to int
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

# Union all edges
for u, v, _ in edges:
    union(int(u), int(v))

# Check if all vertices are in the same set
root = find(0)
all_connected = all(find(i) == root for i in range(X.shape[0]))
assert all_connected, "MST is not connected"

print("\nAll MST properties verified successfully!")

# Calculate some statistics about the edges
weights = [w for _, _, w in edges]
print(f"\nEdge weight statistics:")
print(f"  Minimum weight: {min(weights):.4f}")
print(f"  Maximum weight: {max(weights):.4f}")
print(f"  Average weight: {sum(weights)/len(weights):.4f}")

print("\nTest completed successfully!") 