# PyPAMM Documentation

Welcome to the PyPAMM documentation. PyPAMM is a Python implementation of the Probabilistic Analysis of Molecular Motifs (PAMM) method, originally developed for analyzing molecular dynamics simulations.

## Contents

- [Examples Documentation](examples.md): Detailed documentation of the example scripts
- [API Reference](api.md): API reference for the PyPAMM package

## About PyPAMM

PyPAMM provides efficient, Cython-accelerated implementations of the core PAMM algorithms:

- **Grid Selection**: Implements the min-max algorithm for selecting representative grid points from high-dimensional data
- **Neighbor Graph Construction**: Builds k-nearest neighbor graphs with various distance metrics
- **Quick Shift Clustering**: Implements the Quick Shift algorithm for mode-seeking clustering
- **Minimum Spanning Tree**: Constructs MSTs for efficient data representation and analysis

## Getting Started

To get started with PyPAMM, check out the [examples documentation](examples.md) for detailed examples of how to use the package.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pypamm.git
cd pypamm

# Install with Poetry
poetry install
```

## Running the Examples

```bash
# From the root directory of the repository
poetry run python examples/grid_selection_example.py
poetry run python examples/neighbor_graph_example.py
poetry run python examples/quick_shift_example.py
poetry run python examples/mst_example.py
poetry run python examples/pipeline_example.py
```
