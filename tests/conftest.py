import numpy as np
import pytest


@pytest.fixture
def random_2d_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    return X


@pytest.fixture
def simple_data_voronoi():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]], dtype=np.float64)
    Y = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    wj = np.ones(4, dtype=np.float64)
    idxgrid = np.array([0, 3], dtype=np.int32)
    return X, Y, wj, idxgrid


@pytest.fixture
def simple_data():
    """Simple dataset with known structure for predictable tests."""
    # Create a grid of points in 2D
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points
