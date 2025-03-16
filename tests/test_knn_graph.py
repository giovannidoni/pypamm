import numpy as np
import pytest

# Import the function from the wrapper module
from pypamm.neighbor_graph_wrapper import build_knn_graph


# Fixtures for common test data
@pytest.fixture
def random_data():
    """Generate random data for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(20, 3)  # 20 points in 3D


@pytest.fixture
def simple_data():
    """Simple dataset with known structure for predictable tests."""
    # Create a grid of points in 2D
    data = np.array(
        [
            [0.0, 0.0],  # Origin
            [1.0, 0.0],  # Right
            [0.0, 1.0],  # Top
            [1.0, 1.0],  # Top-right
            [0.5, 0.5],  # Center
        ],
        dtype=np.float64,
    )
    return data


# Test basic functionality
def test_build_knn_graph_basic(random_data):
    """Test basic functionality of build_knn_graph with default parameters."""
    k = 3
    indices, distances = build_knn_graph(random_data, k)

    # Check shapes
    assert indices.shape == (len(random_data), k)
    assert distances.shape == (len(random_data), k)

    # Check types
    assert indices.dtype == np.int32
    assert distances.dtype == np.float64

    # Check that each point has exactly k neighbors
    for i in range(len(random_data)):
        assert len(indices[i]) == k
        assert len(distances[i]) == k

        # Check that distances are non-negative
        assert np.all(distances[i] >= 0)

        # Check that indices are valid
        assert np.all(indices[i] >= 0)
        assert np.all(indices[i] < len(random_data))


# Test with different metrics
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine"])
def test_different_metrics(random_data, metric):
    """Test that different distance metrics work correctly."""
    k = 3
    indices, distances = build_knn_graph(random_data, k, metric=metric)

    # Verify shapes
    assert indices.shape == (len(random_data), k)
    assert distances.shape == (len(random_data), k)


# Test correctness of nearest neighbor search
def test_neighbor_search_correctness(simple_data):
    """Test that the nearest neighbors are correctly identified."""
    k = 2  # Find 2 nearest neighbors
    indices, distances = build_knn_graph(simple_data, k, include_self=False)

    # For the origin point [0,0], the nearest neighbors should include points at indices 1, 2, or 4
    # (these are the closest points: [1,0], [0,1], and [0.5,0.5])
    origin_neighbors = set(indices[0])
    assert len(origin_neighbors.intersection({1, 2, 4})) == 2

    # For the center point [0.5,0.5], all other points are at equal distance
    # So any 2 of the 4 other points could be neighbors
    center_neighbors = set(indices[4])
    assert len(center_neighbors) == 2
    assert all(idx in [0, 1, 2, 3] for idx in center_neighbors)


# Test include_self parameter
def test_include_self(random_data):
    """Test that include_self parameter works correctly."""
    k = 3

    # With include_self=False (default)
    indices_without_self, _ = build_knn_graph(random_data, k, include_self=False)

    # With include_self=True
    indices_with_self, _ = build_knn_graph(random_data, k, include_self=True)

    # When include_self=True, each point should have itself as the nearest neighbor
    for i in range(len(random_data)):
        assert i in indices_with_self[i]
        assert i not in indices_without_self[i]


# Test error cases
def test_invalid_k(random_data):
    """Test that invalid k values raise appropriate errors."""
    # k must be positive
    with pytest.raises(ValueError):
        build_knn_graph(random_data, 0)

    # k must be less than the number of data points
    with pytest.raises(ValueError):
        build_knn_graph(random_data, len(random_data) + 1)


def test_empty_data():
    """Test behavior with empty data."""
    empty_data = np.zeros((0, 3))
    with pytest.raises(ValueError):
        build_knn_graph(empty_data, 1)
