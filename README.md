# PyPAMM

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

### Continuous Integration

PyPAMM uses GitHub Actions for continuous integration and deployment:

- **Testing**: Automatically runs tests on multiple Python versions and operating systems
- **Building**: Builds wheels for different platforms using cibuildwheel
- **Publishing**: Automatically publishes releases to PyPI when a new tag is pushed

For more information about the CI/CD workflows, see the [GitHub Actions README](.github/README.md).

### Project Structure

- `src/pypamm/`: Core package code
  - `grid_selection.pyx`: Min-max grid selection algorithm
  - `neighbor_graph.pyx`: K-nearest neighbor graph construction
  - `distance_metrics.pyx`: Various distance metrics implementations
- `tests/`: Unit tests
- `.github/workflows/`: CI/CD workflows
- `example.py`: Example usage

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
