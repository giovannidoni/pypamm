.PHONY: clean clean-build clean-pyc clean-test clean-all test build install run-example dev all ruff version-patch version-minor version-major version

# Variables
PYTHON = python
PIP = pip
POETRY = poetry

# Default target - builds and installs the package
all: build install

# Version commands
version:
	@echo "Current version: $$($(POETRY) version -s)"

version-patch:
	$(POETRY) run bump2version patch
	@echo "Version bumped to: $$($(POETRY) version -s)"

version-minor:
	$(POETRY) run bump2version minor
	@echo "Version bumped to: $$($(POETRY) version -s)"

version-major:
	$(POETRY) run bump2version major
	@echo "Version bumped to: $$($(POETRY) version -s)"

# Clean build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr src/*.egg-info

# Clean Python cache files
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.pytest_cache' -exec rm -fr {} +
	find . -name '*.so' -exec rm -f {} +

# Clean Cython generated files
clean-cython:
	find src -name '*.c' -exec rm -f {} +
	find src -name '*.cpp' -exec rm -f {} +
	find src -name '*.o' -exec rm -f {} +
	find src -name '*.so' -exec rm -f {} +

# Clean test artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/

# Clean all
clean-all: clean-build clean-pyc clean-cython clean-test

# Default clean command
clean: clean-all

# Run ruff to format and lint code
ruff:
	$(POETRY) run ruff check --fix .
	$(POETRY) run ruff format .

# Build the project
build: clean-build
	$(POETRY) build

# Install the project
install: build
	$(POETRY) install

# Run tests (depends on build and install)
test: install
	$(POETRY) run pytest

# Run tests with verbose output
test-v: install
	$(POETRY) run pytest -v

plots:
	$(POETRY) run python examples/kde_example.py
	$(POETRY) run python examples/quick_shift_example.py

# Development setup - installs dependencies and the package in development mode
dev: clean
	$(POETRY) install

# Help command
help:
	@echo "all - build and install the package (default)"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-cython - remove Cython generated files"
	@echo "clean-test - remove test artifacts"
	@echo "clean-all - remove all artifacts"
	@echo "clean - alias for clean-all"
	@echo "ruff - format and lint code using ruff"
	@echo "build - build the package (depends on clean-build)"
	@echo "install - install the package (depends on build)"
	@echo "test - run tests (depends on install)"
	@echo "test-v - run tests with verbose output (depends on install)"
	@echo "run-example - run the example script (depends on install)"
	@echo "dev - setup development environment"
	@echo "version - show current version"
	@echo "version-patch - bump patch version (0.1.0 -> 0.1.1)"
	@echo "version-minor - bump minor version (0.1.0 -> 0.2.0)"
	@echo "version-major - bump major version (0.1.0 -> 1.0.0)"
