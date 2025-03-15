#!/usr/bin/env python
"""
Dataset Generation Example

This script demonstrates how to use the data generator with YAML configuration files
to create datasets for PyPAMM examples.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset, list_available_configs, load_config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate datasets for PyPAMM examples")
    parser.add_argument("config", nargs="?", default=None, help="Configuration name or path to YAML file")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    parser.add_argument(
        "--output", "-o", default=None, help="Output file path for the generated dataset (numpy .npy format)"
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the generated dataset")
    parser.add_argument("--save-plot", "-p", default=None, help="Save the visualization to the specified file")

    args = parser.parse_args()

    # List available configurations if requested
    if args.list:
        configs = list_available_configs()
        if configs:
            print("Available configurations:")
            for config_name in configs:
                config = load_config(config_name)
                print(f"  - {config_name}: {config.get('description', 'No description')}")
        else:
            print("No configurations found.")
        return

    # Check if a configuration was specified
    if args.config is None:
        print("Error: No configuration specified.")
        print("Use --list to see available configurations or specify a configuration name or file path.")
        return

    # Load the configuration
    try:
        if os.path.exists(args.config):
            print(f"Loading configuration from file: {args.config}")
            config = args.config
        else:
            print(f"Loading predefined configuration: {args.config}")
            config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Generate the dataset
    try:
        X = generate_dataset(config)
        print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return

    # Save the dataset if requested
    if args.output:
        try:
            np.save(args.output, X)
            print(f"Dataset saved to {args.output}")
        except Exception as e:
            print(f"Error saving dataset: {e}")

    # Visualize the dataset if requested
    if args.visualize or args.save_plot:
        try:
            plt.figure(figsize=(10, 8))
            plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
            plt.title("Generated Dataset")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True, alpha=0.3)

            if args.save_plot:
                plt.savefig(args.save_plot)
                print(f"Visualization saved to {args.save_plot}")

            if args.visualize:
                plt.show()
        except Exception as e:
            print(f"Error visualizing dataset: {e}")


if __name__ == "__main__":
    main()
