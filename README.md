# PyPAMM

[![Python Package](https://github.com/giovannidoni/pypamm/actions/workflows/python-package.yml/badge.svg)](https://github.com/giovannidoni/pypamm/actions/workflows/python-package.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/giovannidoni/pypamm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyPAMM is a Python implementation of the Probabilistic Analysis of Molecular Motifs (PAMM) method, originally developed for analyzing molecular dynamics simulations. This package provides efficient, Cython-accelerated implementations of the core PAMM algorithms.

## About PAMM

PAMM (Probabilistic Analysis of Molecular Motifs) is clustering algorithm originally develop for the analusys of molecular simulations:

- Identifies recurring patterns or "motifs" in molecular dynamics data
- Uses density-based clustering to find natural groupings in high-dimensional data
- Provides a probabilistic framework for classifying new observations
- Helps in understanding complex molecular systems by reducing them to a set of interpretable states

This Python port aims to make PAMM more accessible to the scientific community by providing a user-friendly interface while maintaining high performance through Cython optimizations.

The original PAMM method was developed by the Laboratory of Computational Science and Modeling (COSMO) at EPFL and implemented in Fortran. The original repository can be found at [https://github.com/lab-cosmo/pamm](https://github.com/lab-cosmo/pamm).

## Features

- **Grid Selection**: Implements the min-max algorithm for selecting representative grid points from high-dimensional data
- **Neighbor Graph Construction**: Builds k-nearest neighbor graphs with various distance metrics
- **Quick Shift Clustering**: Implements the Quick Shift algorithm for mode-seeking clustering
  - **Enhanced with Graph-based Optimization**: Significantly faster for large datasets using pre-computed neighbor graphs
- **Minimum Spanning Tree**: Constructs MSTs for efficient data representation and analysis
- **High Performance**: Core algorithms implemented in Cython for speed
- **Memory Efficient**: Uses typed memory views to minimize memory overhead and avoid unnecessary data copies
- **No GIL**: Critical sections release the Python GIL for true parallel execution
- **Multiple Distance Metrics**: Supports Euclidean, Manhattan, Chebyshev, Cosine, Mahalanobis, and Minkowski distances
- **Flexible API**: Simple interface for integration with existing Python workflows
- **Modern Python Support**: Fully compatible with Python 3.12 and NumPy 2.0.0

## Installation

### Prerequisites

- Python 3.10 or higher
- NumPy 2.0.0 or higher
- Cython 3.0.0 or higher
- SciPy 1.12.0 or higher

### Using Poetry (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/pypamm.git
cd pypamm

# Install with Poetry
poetry install
```

### Using pip

```bash
pip install git+https://github.com/yourusername/pypamm.git
```

## Usage Examples

PyPAMM includes several example scripts that demonstrate how to use the various algorithms and functions in the package. These examples are located in the `examples/` directory.

### Running the Examples

```bash
# From the root directory of the repository
poetry run python examples/grid_selection_example.py
poetry run python examples/neighbor_graph_example.py
poetry run python examples/quick_shift_example.py
poetry run python examples/quick_shift_graph_example.py
poetry run python examples/mst_example.py
poetry run python examples/pipeline_example.py
poetry run python examples/pamm_clustering_example.py
```

### Available Examples

- **Grid Selection**: Demonstrates how to select a subset of points from a dataset based on a grid
- **Neighbor Graph**: Shows how to build different types of neighborhood graphs
- **Quick Shift Clustering**: Illustrates the use of the Quick Shift algorithm for clustering
  - **Graph-based Quick Shift**: Shows how to use pre-computed neighbor graphs for faster clustering
- **Minimum Spanning Tree (MST)**: Demonstrates how to build and use MSTs
- **Complete Pipeline**: Shows how to combine multiple algorithms in a data analysis pipeline
- **PAMM Clustering**: Demonstrates the complete PAMM algorithm with bootstrapping and graph-based optimization

For more detailed information about the examples, see the [Examples Documentation](docs/examples.md).

## Technical Details

### Array Layout

PyPAMM expects C-contiguous arrays (row-major order) as input. This is the default layout for NumPy arrays, so in most cases you don't need to modify your arrays. In the rare case where you're working with Fortran-contiguous arrays (column-major order), you may need to convert them:

```python
import numpy as np

# Convert Fortran-contiguous to C-contiguous
c_array = np.ascontiguousarray(f_array)
```

### Memory Efficiency

PyPAMM uses Cython's typed memory views extensively to reduce memory overhead and avoid unnecessary copying of data. This allows us to:

- Process large datasets efficiently
- Minimize memory footprint during computation
- Pass data by reference rather than by value

### Release of GIL

Performance-critical sections of the code use the `nogil` context to release the Python Global Interpreter Lock (GIL), enabling true parallel execution when used with Python's multiprocessing or multithreading.

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/pypamm.git
cd pypamm

# Install development dependencies
poetry install

# Run tests
poetry run pytest
```

### Project Structure

- `src/pypamm/`: Core package code
  - `grid_selection.pyx`: Min-max grid selection algorithm
  - `neighbor_graph.pyx`: K-nearest neighbor graph construction
  - `distance_metrics.pyx`: Various distance metrics implementations
  - `quick_shift.pyx`: Quick Shift clustering algorithm
  - `mst.pyx`: Minimum Spanning Tree construction
- `examples/`: Example scripts demonstrating package usage
- `docs/`: Documentation
- `tests/`: Unit tests
- `.github/workflows/`: CI/CD workflows

### Code Formatting and Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting, configured with a line length of 120 characters.

We use [pre-commit](https://pre-commit.com/) to automatically run Ruff before each commit. To set up pre-commit:

1. Install the development dependencies:
   ```bash
   poetry install --with dev
   ```

2. Install the pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

3. Now, Ruff will automatically format and lint your code before each commit.

You can also run the hooks manually on all files:
```bash
poetry run pre-commit run --all-files
```

Or run Ruff directly:
```bash
# Format code
poetry run ruff format .

# Lint code
poetry run ruff check --fix .
```

## Continuous Integration and Deployment

PyPAMM uses GitHub Actions for continuous integration and deployment.

### GitHub Actions Workflows

#### Python Package (`python-package.yml`)

This workflow runs tests on multiple Python versions and operating systems.

**Triggered by:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
- **test**: Runs tests on Ubuntu and macOS with Python 3.10, 3.11, and 3.12
- **build**: Builds the package using Poetry and uploads the artifacts

### Publishing to PyPI

To publish a new release to PyPI:

1. Update the version in `pyproject.toml`
2. Commit the changes
3. Create and push a new tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

### Required Secrets

For the PyPI publishing to work, you need to add the following secret to your GitHub repository:

- `PYPI_API_TOKEN`: A PyPI API token with upload permissions

You can add this secret in your GitHub repository settings under "Settings > Secrets and variables > Actions".

## Documentation

- [Examples Documentation](docs/examples.md): Detailed documentation of the example scripts
- [API Reference](docs/api.md): API reference for the PyPAMM package

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use PyPAMM in your research, please cite:

```
@article{pamm-original,
  title={Probabilistic analysis of molecular motifs in biomolecular structures},
  author={Gasparotto, Piero and Ceriotti, Michele},
  journal={The Journal of Chemical Physics},
  volume={140},
  number={23},
  year={2014},
  publisher={AIP Publishing}
}
```

## Acknowledgments

This package is a Python port of the original PAMM method developed by Piero Gasparotto and Michele Ceriotti at the Laboratory of Computational Science and Modeling (COSMO) at EPFL. The original Fortran implementation can be found at [https://github.com/lab-cosmo/pamm](https://github.com/lab-cosmo/pamm). We thank them for their pioneering work in this field.

## Versioning

PyPAMM follows [Semantic Versioning](https://semver.org/) with version numbers in the format MAJOR.MINOR.PATCH:

- **MAJOR** version: Incremented for incompatible API changes
- **MINOR** version: Incremented for added functionality in a backward-compatible manner
- **PATCH** version: Incremented for backward-compatible bug fixes

### Version Management

The project includes commands to manage versions using the industry-standard `bump2version` tool:

```bash
# Show current version
make version

# Bump versions
make version-patch  # 0.1.0 -> 0.1.1
make version-minor  # 0.1.0 -> 0.2.0
make version-major  # 0.1.0 -> 1.0.0
```

When you run these commands:
1. The version number is updated in all relevant files
2. A git commit is created with the version change
3. A git tag is created (e.g., `v0.1.1`)

You can then push the changes and tags to your repository:

```bash
git push && git push --tags
```

To use a specific version of the package in your project, specify it in your dependency requirements:

```bash
poetry add pypamm@^0.1.0  # Compatible with 0.1.x
```
