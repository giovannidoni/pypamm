.PHONY: clean clean-build clean-pyc clean-test clean-all test build install

# Variables
PYTHON = python
PIP = pip
POETRY = poetry

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

# Run tests
test:
	$(POETRY) run pytest

# Build the project
build:
	$(POETRY) build

# Install the project
install:
	$(POETRY) install

# Run the example
run-example:
	$(POETRY) run python example.py

# Help command
help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-cython - remove Cython generated files"
	@echo "clean-test - remove test artifacts"
	@echo "clean-all - remove all artifacts"
	@echo "clean - alias for clean-all"
	@echo "test - run tests"
	@echo "build - build the package"
	@echo "install - install the package"
	@echo "run-example - run the example script" 