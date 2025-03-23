import numpy as np
import pytest

from pypamm.lib.distance import py_calculate_distance


# Fixtures for common test data
@pytest.fixture
def random_vectors():
    """Generate random vectors for testing distance functions."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(10, 3)  # 10 vectors in 3D


@pytest.fixture
def simple_vectors():
    """Simple vectors with known distances for predictable tests."""
    return np.array(
        [
            [0.0, 0.0, 0.0],  # Origin
            [1.0, 0.0, 0.0],  # Unit vector along x
            [0.0, 1.0, 0.0],  # Unit vector along y
            [0.0, 0.0, 1.0],  # Unit vector along z
            [1.0, 1.0, 1.0],  # (1,1,1) vector
        ],
        dtype=np.float64,
    )


# Test distance calculations
def test_euclidean_distance(simple_vectors):
    """Test Euclidean distance calculations."""
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)

    # Distance from origin to (1,0,0) should be 1
    dist_a_b = py_calculate_distance("euclidean", a_view, b_view)
    assert np.isclose(dist_a_b, 1.0)

    # Distance from origin to (1,1,1) should be 3 (squared Euclidean)
    dist_a_c = py_calculate_distance("euclidean", a_view, c_view)
    assert np.isclose(dist_a_c, 3.0)

    # Distance from (1,0,0) to (1,1,1) should be 2 (squared Euclidean)
    dist_b_c = py_calculate_distance("euclidean", b_view, c_view)
    assert np.isclose(dist_b_c, 2.0)


def test_manhattan_distance(simple_vectors):
    """Test Manhattan distance calculations."""
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)

    # Distance from origin to (1,0,0) should be 1
    dist_a_b = py_calculate_distance("manhattan", a_view, b_view)
    assert np.isclose(dist_a_b, 1.0)

    # Distance from origin to (1,1,1) should be 3
    dist_a_c = py_calculate_distance("manhattan", a_view, c_view)
    assert np.isclose(dist_a_c, 3.0)

    # Distance from (1,0,0) to (1,1,1) should be 2
    dist_b_c = py_calculate_distance("manhattan", b_view, c_view)
    assert np.isclose(dist_b_c, 2.0)


def test_chebyshev_distance(simple_vectors):
    """Test Chebyshev distance calculations."""
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)

    # Distance from origin to (1,0,0) should be 1
    dist_a_b = py_calculate_distance("chebyshev", a_view, b_view)
    assert np.isclose(dist_a_b, 1.0)

    # Distance from origin to (1,1,1) should be 1 (max of |1|, |1|, |1|)
    dist_a_c = py_calculate_distance("chebyshev", a_view, c_view)
    assert np.isclose(dist_a_c, 1.0)

    # Distance from (1,0,0) to (1,1,1) should be 1 (max of |0|, |1|, |1|)
    dist_b_c = py_calculate_distance("chebyshev", b_view, c_view)
    assert np.isclose(dist_b_c, 1.0)


def test_cosine_distance(simple_vectors):
    """Test Cosine distance calculations."""
    # Calculate distances between vectors
    a = simple_vectors[1]  # (1,0,0)
    b = simple_vectors[2]  # (0,1,0)
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)

    # Distance between orthogonal vectors should be 1
    dist_a_b = py_calculate_distance("cosine", a_view, b_view)
    assert np.isclose(dist_a_b, 1.0)

    # Distance from (1,0,0) to (1,1,1) should be 1 - 1/sqrt(3) ≈ 0.4226
    dist_a_c = py_calculate_distance("cosine", a_view, c_view)
    assert np.isclose(dist_a_c, 1.0 - 1.0 / np.sqrt(3.0))


def test_mahalanobis_distance(simple_vectors):
    """Test Mahalanobis distance calculations."""
    # Use identity matrix for simplicity (reduces to squared Euclidean)
    inv_cov_mat = np.eye(3)

    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov_mat)

    # With identity matrix, should be same as squared Euclidean
    dist_a_b = py_calculate_distance("mahalanobis", a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)

    dist_a_c = py_calculate_distance("mahalanobis", a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 3.0)


def test_minkowski_distance(simple_vectors):
    """Test Minkowski distance calculations."""
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    c = simple_vectors[4]  # (1,1,1)

    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    c_view = np.ascontiguousarray(c)

    # With p=1, should be same as Manhattan
    dist_a_c = py_calculate_distance("minkowski", a_view, c_view, k=1.0)
    assert np.isclose(dist_a_c, 3.0)

    # With p=2, should be same as Euclidean (squared)
    dist_a_c = py_calculate_distance("minkowski", a_view, c_view, k=2.0)
    expected = (1.0 + 1.0 + 1.0) ** (1.0 / 2.0)
    assert np.isclose(dist_a_c, expected)

    # With p=3
    dist_a_c = py_calculate_distance("minkowski", a_view, c_view, k=3.0)
    expected = (1.0**3 + 1.0**3 + 1.0**3) ** (1.0 / 3.0)
    print(dist_a_c, expected)
    assert np.isclose(dist_a_c, expected)


def test_invalid_metric():
    """Test error handling for invalid metrics."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported metric"):
        py_calculate_distance("invalid_metric", a, b)


def test_mahalanobis_without_inv_cov():
    """Test Mahalanobis without inv_cov raises error."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        py_calculate_distance("mahalanobis", a, b)
