#!/usr/bin/env python
"""
Kernel Density Estimation (KDE) Example

This example demonstrates how to use the KDE functionality in pypamm
to estimate probability density functions from data points. KDE is a
non-parametric way to estimate the probability density function of a
random variable based on observed data.

The example uses a YAML configuration file for all parameters.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset

from pypamm.density.kde import compute_kde

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    import yaml

    config = yaml.safe_load(f)

# Generate the dataset
X = generate_dataset(config)

# Get KDE parameters from config
kde_config = config["kde"]
bandwidth = kde_config["bandwidth"]

# Apply KDE
start_time = time.time()
density = compute_kde(X, X, bandwidth)
end_time = time.time()
kde_time = end_time - start_time

print("\nKDE Computation:")
print(f"  Computed density for {len(X)} points")
print(f"  Bandwidth: {bandwidth}")
print(f"  Computation time: {kde_time:.4f} seconds")

# Set up the figure
viz_config = config["visualization"]
fig, ax = plt.subplots(figsize=tuple(viz_config["figsize"]))

# Create a grid for density visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Compute density on the grid
grid_density = compute_kde(grid_points, X, bandwidth)
grid_density = grid_density.reshape(xx.shape)

# Plot the density as a contour plot
contour = ax.contourf(xx, yy, grid_density, levels=15, cmap=viz_config["cmap"], alpha=0.8)

# Plot the data points
scatter = ax.scatter(
    X[:, 0],
    X[:, 1],
    c="white",
    s=viz_config["point_size"] * 0.5,
    edgecolors="black",
    linewidths=0.5,
    alpha=viz_config["alpha"],
)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Density")

# Add title and labels
ax.set_title(f"Kernel Density Estimation\n(bandwidth={bandwidth}, {kde_time:.2f}s)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# Adjust layout and save figure
plt.tight_layout()
output_path = viz_config["kde_output_path"]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")

# Print explanation of KDE
print("\nExplanation of Kernel Density Estimation (KDE):")
print("---------------------------------------------")
print("KDE is a non-parametric method to estimate the probability density function of a random variable.")
print("It works by:")
print("1. Placing a kernel (usually a Gaussian) on each data point")
print("2. Summing these kernels to create a smooth density estimate")
print("3. Using bandwidth to control the smoothness of the estimate")

print("\nKey Parameters:")
print(f"- bandwidth: {bandwidth} (Controls the smoothness of the density estimation)")
print("  - Smaller bandwidth: More detail, but potentially noisy")
print("  - Larger bandwidth: Smoother, but may miss important features")

print("\nApplications in PAMM:")
print("- KDE provides the density estimate used by QuickShift for mode-seeking")
print("- It helps identify regions of high density where clusters are centered")
print("- The bandwidth parameter is critical for proper cluster identification")

print("\nExample completed successfully!")
