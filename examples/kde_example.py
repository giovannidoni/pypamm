#!/usr/bin/env python
"""
Kernel Density Estimation (KDE) Example

This example demonstrates how to use the KDE functionality from the PyPAMM library
to estimate probability density functions from data points. KDE is a non-parametric
method to estimate the probability density function of a random variable based on
a finite data sample.

The example uses datasets defined in the example_config.yaml file and visualizes
the density estimation results using matplotlib.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset, load_config
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from pypamm.density.kde import compute_kde
from pypamm.grid_selection_wrapper import select_grid_points

# Load configuration from YAML file
config = load_config("example_config")

# Get visualization settings
viz_config = config["visualization"]
output_dir = viz_config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# Use configured datasets and bandwidths
config = config["examples"]["kde_example"]
kde_config = config["kde"]
datasets = config.get("datasets", ["dataset_default"])
bandwidths = kde_config.get("bandwidths", [0.1])
ngrid = kde_config.get("ngrid", 100)  # Get grid size from config

# Create figure layout based on number of datasets and bandwidths
n_datasets = len(datasets)
n_bandwidths = len(bandwidths) + 1
fig = plt.figure(figsize=(5 * n_bandwidths, 5 * n_datasets))
gs = GridSpec(n_datasets, n_bandwidths, figure=fig)

for i, dataset_name in enumerate(datasets):
    print(f"\nProcessing dataset: {dataset_name}")

    # Generate dataset
    X = generate_dataset(dataset_name, "data_config")
    print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

    # Select grid points using PyPAMM's grid_selection function
    print(f"Selecting {ngrid} grid points for KDE evaluation...")
    grid_points_idx, _ = select_grid_points(X, ngrid, metric="euclidean")
    grid_points = X[grid_points_idx]

    # Create a regular grid for visualization (always in 2D)
    # For higher dimensions, we'll project onto the first two dimensions

    x_min = X[:, 1].min()
    x_max = X[:, 0].max()

    if X.shape[1] >= 2:
        y_min = X[:, 1].min() - 0.1
        y_max = X[:, 1].max() + 0.1
    else:
        # For 1D data, create a dummy second dimension
        y_min, y_max = -0.1, 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # For 1D data, set second dimension to 0
    if X.shape[1] == 1:
        viz_grid = np.c_[xx.ravel(), np.zeros_like(xx.ravel())]
    else:
        # For 2D+ data, use first two dimensions for visualization grid
        viz_grid = np.c_[xx.ravel(), yy.ravel()]

        # If data has more than 2 dimensions, pad with zeros
        if X.shape[1] > 2:
            padding = np.zeros((viz_grid.shape[0], X.shape[1] - 2))
            viz_grid = np.hstack([viz_grid, padding])

    for j, bandwidth in enumerate(bandwidths + ["adaptive"]):
        print(f"  Computing KDE with bandwidth={bandwidth}")

        # Create subplot
        ax = fig.add_subplot(gs[i, j])

        # Compute KDE on the grid points
        if bandwidth == "adaptive":
            density = compute_kde(X, grid_points, adaptive=True)
        else:
            density = compute_kde(X, grid_points, constant_bandwidth=bandwidth, adaptive=False)

        # Compute KDE on the visualization grid
        viz_density = compute_kde(X, viz_grid, constant_bandwidth=bandwidth, adaptive=True)

        density_grid = viz_density.reshape(xx.shape)

        # Plot the data points (first two dimensions only)
        if X.shape[1] == 1:
            # For 1D data, plot points along x-axis with y=0
            ax.scatter(X[:, 0], np.zeros(X.shape[0]), s=20, alpha=0.5, edgecolor="k")
        else:
            # For 2D+ data, plot first two dimensions
            ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.5, edgecolor="k")

        ax.scatter(grid_points[:, 0], grid_points[:, 1], s=30, color="red", alpha=0.7, marker="x", label="Grid Points")

        # Plot density contours
        contour = ax.contourf(xx, yy, density_grid, cmap=cm.viridis, alpha=0.8)
        plt.colorbar(contour, ax=ax, label="Density")

        # Add legend
        ax.legend(loc="upper right")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set title and labels
        dim_info = f"{X.shape[1]}D" if X.shape[1] > 2 else ""
        if X.shape[1] > 2:
            title = f"KDE {dim_info} (bandwidth={bandwidth}, ngrid={ngrid})\nProjected to first 2 dimensions"
        else:
            title = f"KDE (bandwidth={bandwidth}, ngrid={ngrid})"

        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2" if X.shape[1] > 1 else "")

        # Add dataset name as row title if first column
        if j == 0:
            ax.set_ylabel(f"Dataset: {dataset_name}\n{ax.get_ylabel()}", fontsize=12)

# Adjust layout
plt.tight_layout()  # Make room for the suptitle

# Save the figure
output_path = f"{output_dir}/kde_example.png"
plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")
print(f"\nVisualization saved to {output_path}")
