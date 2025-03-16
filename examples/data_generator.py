#!/usr/bin/env python
"""
Data Generator for PyPAMM Examples

This module provides functions to generate synthetic datasets for
demonstrating PyPAMM algorithms. It supports various dataset types
including clusters, rings, lines, and noise.
"""

import os

import numpy as np
import yaml


def generate_dataset(config: dict | str) -> np.ndarray:
    """
    Generate a synthetic dataset based on configuration parameters.

    Parameters:
    -----------
    config : dict or str
        Either a dictionary with configuration parameters or a path to a YAML config file.
        The configuration should specify dataset parameters including:
        - random_seed: int, seed for reproducibility
        - components: list of component configurations, each with:
          - type: str, type of component ('cluster', 'ring', 'line', 'noise', 'grid')
          - n_points: int, number of points in this component
          - center: list, center coordinates
          - scale: float or list, scale/spread of the component
          - additional component-specific parameters

    Returns:
    --------
    X : np.ndarray
        Generated dataset with shape (n_points, n_dimensions)
    """
    # Load config from file if a string is provided
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    # Set random seed for reproducibility
    if "random_seed" in config:
        np.random.seed(config["random_seed"])

    # Initialize empty dataset
    data_components = []

    # Generate each component
    for component in config["components"]:
        component_type = component["type"]
        n_points = component["n_points"]

        if component_type == "cluster":
            # Gaussian cluster
            center = np.array(component["center"])
            if isinstance(component["scale"], list):
                scale = np.array(component["scale"])
            else:
                scale = component["scale"]

            if "dimensions" in component:
                dims = component["dimensions"]
            else:
                dims = len(center)

            if isinstance(scale, np.ndarray):
                points = np.random.randn(n_points, dims) * scale[:, np.newaxis].T + center
            else:
                points = np.random.randn(n_points, dims) * scale + center

        elif component_type == "ring":
            # Ring-shaped cluster
            center = np.array(component["center"])
            radius = component["radius"]
            thickness = component.get("thickness", 0.1)

            theta = np.random.uniform(0, 2 * np.pi, n_points)
            r = np.random.normal(radius, thickness, n_points)

            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]

            points = np.vstack([x, y]).T

        elif component_type == "line":
            # Line-shaped cluster
            start = np.array(component["start"])
            end = np.array(component["end"])
            thickness = component.get("thickness", 0.1)

            # Generate points along the line
            t = np.random.rand(n_points, 1)
            line_points = start + t * (end - start)

            # Add perpendicular noise
            direction = end - start
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)

            noise = np.random.randn(n_points, 1) * thickness
            points = line_points + noise * perpendicular

        elif component_type == "grid":
            # Grid of clusters
            grid_size = component["grid_size"]
            min_coords = component.get("min_coords", [0, 0])
            max_coords = component.get("max_coords", [10, 10])
            cluster_scale = component.get("cluster_scale", 0.1)

            # Calculate the number of points per grid cell
            points_per_cell = n_points // (grid_size[0] * grid_size[1])
            if points_per_cell < 1:
                points_per_cell = 1

            # Generate grid centers
            x_centers = np.linspace(min_coords[0], max_coords[0], grid_size[0])
            y_centers = np.linspace(min_coords[1], max_coords[1], grid_size[1])

            # Generate points for each grid cell
            grid_points = []
            for x in x_centers:
                for y in y_centers:
                    # Create a small cluster at each grid point
                    cluster = np.random.randn(points_per_cell, 2) * cluster_scale + np.array([x, y])
                    grid_points.append(cluster)

            points = np.vstack(grid_points)

        elif component_type == "noise":
            # Uniform noise
            min_val = component.get("min", 0)
            max_val = component.get("max", 10)
            dims = component.get("dimensions", 2)

            points = np.random.uniform(min_val, max_val, (n_points, dims))

        else:
            raise ValueError(f"Unknown component type: {component_type}")

        data_components.append(points)

    # Combine all components
    X = np.vstack(data_components)

    # Shuffle the data if requested
    if config.get("shuffle", True):
        np.random.shuffle(X)

    return X


def load_config(name: str) -> dict:
    """
    Load a predefined configuration by name.

    Parameters:
    -----------
    name : str
        Name of the predefined configuration

    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    # Check if it's a path to a YAML file
    if os.path.exists(name) and (name.endswith(".yaml") or name.endswith(".yml")):
        with open(name) as f:
            return yaml.safe_load(f)

    # Check if it's a predefined configuration name
    config_dir = os.path.join(os.path.dirname(__file__), "configs")

    # Try with .yaml extension
    config_path = os.path.join(config_dir, f"{name}.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Try with .yml extension
    config_path = os.path.join(config_dir, f"{name}.yml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    raise ValueError(f"Configuration '{name}' not found")


def save_config(config: dict, path: str) -> None:
    """
    Save a configuration to a YAML file.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    path : str
        Path to save the configuration
    """
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def list_available_configs() -> list[str]:
    """
    List all available predefined configurations.

    Returns:
    --------
    config_names : list
        List of available configuration names
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    if not os.path.exists(config_dir):
        return []

    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml") or f.endswith(".yml")]
    config_names = [os.path.splitext(f)[0] for f in config_files]
    return sorted(config_names)


# Example usage
if __name__ == "__main__":
    # List available configurations
    configs = list_available_configs()
    if configs:
        print("Available configurations:")
        for config_name in configs:
            config = load_config(config_name)
            print(f"  - {config_name}: {config.get('description', 'No description')}")

    # Example configuration
    example_config = {
        "random_seed": 42,
        "shuffle": True,
        "components": [
            {"type": "cluster", "n_points": 200, "center": [2, 2], "scale": 0.5},
            {"type": "ring", "n_points": 150, "center": [6, 6], "radius": 2, "thickness": 0.2},
            {"type": "line", "n_points": 100, "start": [0, 0], "end": [5, 5], "thickness": 0.3},
            {"type": "noise", "n_points": 50, "min": 0, "max": 10, "dimensions": 2},
        ],
        "description": "Example dataset with multiple component types",
    }

    # Save example config to YAML file
    example_config_path = os.path.join(os.path.dirname(__file__), "configs", "example.yaml")
    save_config(example_config, example_config_path)
    print(f"Saved example configuration to {example_config_path}")

    # Generate and visualize the dataset
    X = generate_dataset(example_config)
    print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")

    # Optionally visualize
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
        plt.title("Example Generated Dataset")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.savefig("example_dataset.png")
        print("Saved visualization to 'example_dataset.png'")
    except ImportError:
        print("Matplotlib not available for visualization")
