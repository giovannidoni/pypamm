# PyPAMM

[![Python Package](https://github.com/giovannidoni/pypamm/actions/workflows/python-package.yml/badge.svg)](https://github.com/giovannidoni/pypamm/actions/workflows/python-package.yml)
[![Build Wheels](https://github.com/giovannidoni/pypamm/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/giovannidoni/pypamm/actions/workflows/build-wheels.yml)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/giovannidoni/pypamm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyPAMM is a Python implementation of the Probabilistic Analysis of Molecular Motifs (PAMM) method, originally developed for analyzing molecular dynamics simulations. This package provides efficient, Cython-accelerated implementations of the core PAMM algorithms.

## About PAMM

PAMM (Probabilistic Analysis of Molecular Motifs) is a statistical analysis method that:

- Identifies recurring patterns or "motifs" in molecular dynamics data
- Uses density-based clustering to find natural groupings in high-dimensional data
- Provides a probabilistic framework for classifying new observations
- Helps in understanding complex molecular systems by reducing them to a set of interpretable states

This Python port aims to make PAMM more accessible to the scientific community by providing a user-friendly interface while maintaining high performance through Cython optimizations.

The original PAMM method was developed by the Laboratory of Computational Science and Modeling (COSMO) at EPFL and implemented in Fortran. The original repository can be found at [https://github.com/lab-cosmo/pamm](https://github.com/lab-cosmo/pamm).

## Features

- **Grid Selection**: Implements the min-max algorithm for selecting representative grid points from high-dimensional data
- **Neighbor Graph Construction**: Builds k-nearest neighbor graphs with various distance metrics
- **High Performance**: Core algorithms implemented in Cython for speed
- **Multiple Distance Metrics**: Supports Euclidean, Manhattan, Chebyshev, Cosine, Mahalanobis, and Minkowski distances
- **Flexible API**: Simple interface for integration with existing Python workflows

## Installation

### Prerequisites

- Python 3.12 or higher
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
- `tests/`: Unit tests
- `.github/workflows/`: CI/CD workflows

## Continuous Integration and Deployment

PyPAMM uses GitHub Actions for continuous integration and deployment.

### GitHub Actions Workflows

#### 1. Python Package (`python-package.yml`)

This workflow runs tests on multiple Python versions and operating systems.

**Triggered by:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
- **test**: Runs tests on Ubuntu and macOS with Python 3.9, 3.10, 3.11, and 3.12
- **build**: Builds the package using Poetry and uploads the artifacts

#### 2. Build Wheels (`build-wheels.yml`)

This workflow builds wheels for different platforms using cibuildwheel.

**Triggered by:**
- Push to `main` branch
- Push of tags starting with `v`
- Pull requests to `main` branch

**Jobs:**
- **build_wheels**: Builds wheels for Ubuntu, Windows, and macOS with Python 3.9, 3.10, 3.11, and 3.12
- **build_sdist**: Builds a source distribution
- **publish**: Publishes the built distributions to PyPI (only when a tag is pushed)

### Publishing to PyPI

To publish a new release to PyPI:

1. Update the version in `pyproject.toml`
2. Commit the changes
3. Create and push a new tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

This will trigger the `build-wheels.yml` workflow, which will build wheels for all platforms and publish them to PyPI.

### Required Secrets

For the PyPI publishing to work, you need to add the following secret to your GitHub repository:

- `PYPI_API_TOKEN`: A PyPI API token with upload permissions

You can add this secret in your GitHub repository settings under "Settings > Secrets and variables > Actions".

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

And also cite this Python implementation:

```
@software{pypamm,
  title={PyPAMM: Python Implementation of Probabilistic Analysis of Molecular Motifs},
  author={Your Name},
  url={https://github.com/yourusername/pypamm},
  year={2023}
}
```

## Acknowledgments

This package is a Python port of the original PAMM method developed by Piero Gasparotto and Michele Ceriotti at the Laboratory of Computational Science and Modeling (COSMO) at EPFL. The original Fortran implementation can be found at [https://github.com/lab-cosmo/pamm](https://github.com/lab-cosmo/pamm). We thank them for their pioneering work in this field.
