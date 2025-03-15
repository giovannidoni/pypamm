# GitHub Actions Workflows for PyPAMM

This directory contains GitHub Actions workflows for testing and building the PyPAMM package.

## Workflows

### 1. Python Package (`python-package.yml`)

This workflow runs tests on multiple Python versions and operating systems.

**Triggered by:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
- **test**: Runs tests on Ubuntu and macOS with Python 3.9, 3.10, 3.11, and 3.12
- **build**: Builds the package using Poetry and uploads the artifacts

### 2. Build Wheels (`build-wheels.yml`)

This workflow builds wheels for different platforms using cibuildwheel.

**Triggered by:**
- Push to `main` branch
- Push of tags starting with `v`
- Pull requests to `main` branch

**Jobs:**
- **build_wheels**: Builds wheels for Ubuntu, Windows, and macOS with Python 3.9, 3.10, 3.11, and 3.12
- **build_sdist**: Builds a source distribution
- **publish**: Publishes the built distributions to PyPI (only when a tag is pushed)

## Publishing to PyPI

To publish a new release to PyPI:

1. Update the version in `pyproject.toml`
2. Commit the changes
3. Create and push a new tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

This will trigger the `build-wheels.yml` workflow, which will build wheels for all platforms and publish them to PyPI.

## Required Secrets

For the PyPI publishing to work, you need to add the following secret to your GitHub repository:

- `PYPI_API_TOKEN`: A PyPI API token with upload permissions

You can add this secret in your GitHub repository settings under "Settings > Secrets and variables > Actions". 