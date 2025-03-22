import numpy as np
import pytest

from pypamm.grid_selection_wrapper import select_grid_points


# Fixtures for common test data
@pytest.fixture
def random_data():
    """Generate random data for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(100, 3)  # 100 points in 3D


@pytest.fixture
def simple_data():
    """Simple dataset with known structure for predictable tests."""
    # Create a grid of points in 2D
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points


# Test basic functionality
def test_select_grid_points_basic(random_data):
    """Test basic functionality of select_grid_points with default parameters."""
    ngrid = 10
    idxgrid, Y = select_grid_points(random_data, ngrid)

    # Check output shapes
    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, random_data.shape[1])

    # Check that indices are within bounds
    assert np.all(idxgrid >= 0)
    assert np.all(idxgrid < random_data.shape[0])

    # Check that Y contains the correct points
    for i, idx in enumerate(idxgrid):
        np.testing.assert_array_equal(Y[i], random_data[idx])


# Test with different metrics
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine"])
def test_different_metrics(random_data, metric):
    """Test select_grid_points with different distance metrics."""
    ngrid = 5
    idxgrid, Y = select_grid_points(random_data, ngrid, metric=metric)

    # Check output shapes
    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, random_data.shape[1])

    # Check that indices are within bounds
    assert np.all(idxgrid >= 0)
    assert np.all(idxgrid < random_data.shape[0])


# Test with Mahalanobis distance
def test_mahalanobis_distance(random_data):
    """Test select_grid_points with Mahalanobis distance."""
    ngrid = 5
    D = random_data.shape[1]

    # Create an identity matrix for simplicity (reduces to Euclidean)
    inv_cov = np.eye(D)

    idxgrid, Y = select_grid_points(random_data, ngrid, metric="mahalanobis", inv_cov=inv_cov)

    # Check output shapes
    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, D)

    # Check that indices are within bounds
    assert np.all(idxgrid >= 0)
    assert np.all(idxgrid < random_data.shape[0])


# Test with Minkowski distance
def test_minkowski_distance(random_data):
    """Test select_grid_points with Minkowski distance."""
    ngrid = 5

    # Create parameter for p=2 (Euclidean)
    p = np.array([[2.0]])

    idxgrid, Y = select_grid_points(random_data, ngrid, metric="minkowski", inv_cov=p)

    # Check output shapes
    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, random_data.shape[1])

    # Check that indices are within bounds
    assert np.all(idxgrid >= 0)
    assert np.all(idxgrid < random_data.shape[0])


# Test error handling
def test_invalid_metric(random_data):
    """Test that an invalid metric raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        select_grid_points(random_data, 5, metric="invalid_metric")


def test_mahalanobis_without_inv_cov(random_data):
    """Test that Mahalanobis without inv_cov raises a ValueError."""
    with pytest.raises(ValueError, match="Must supply inv_cov"):
        select_grid_points(random_data, 5, metric="mahalanobis")


def test_mahalanobis_wrong_shape(random_data):
    """Test that Mahalanobis with wrong inv_cov shape raises a ValueError."""
    D = random_data.shape[1]
    with pytest.raises(ValueError, match="inv_cov must be"):
        select_grid_points(random_data, 5, metric="mahalanobis", inv_cov=np.eye(D + 1))


def test_minkowski_without_param(random_data):
    """Test that Minkowski without parameter raises a ValueError."""
    with pytest.raises(ValueError, match="Must supply a 1x1 array"):
        select_grid_points(random_data, 5, metric="minkowski")


def test_minkowski_wrong_shape(random_data):
    """Test that Minkowski with wrong parameter shape raises a ValueError."""
    with pytest.raises(ValueError, match="inv_cov must be a 1x1 array"):
        select_grid_points(random_data, 5, metric="minkowski", inv_cov=np.array([[1.0, 2.0]]))


# Test algorithm correctness
def test_min_max_algorithm_correctness(simple_data):
    """Test that the min-max algorithm selects points that maximize minimum distance."""
    ngrid = 4
    idxgrid, Y = select_grid_points(simple_data, ngrid)

    # The algorithm should select points that are far apart
    # For our simple grid, we expect points near the corners

    # Calculate pairwise distances between selected points
    min_dist = float("inf")
    for i in range(ngrid):
        for j in range(i + 1, ngrid):
            dist = np.sum((Y[i] - Y[j]) ** 2)  # Squared Euclidean distance
            min_dist = min(min_dist, dist)

    # The minimum distance should be relatively large
    assert min_dist > 0.1, "Selected points are too close together"


# Test edge cases
def test_ngrid_equals_data_size():
    """Test when ngrid equals the number of data points."""
    data = np.random.rand(5, 2)
    idxgrid, Y = select_grid_points(data, 5)

    # Should select all points
    assert len(idxgrid) == 5
    assert len(np.unique(idxgrid)) == 5  # All indices should be unique


def test_comprehensive_ngrid_equals_data_size():
    """Comprehensive test for when ngrid equals the number of data points."""
    # Create a structured dataset with known points
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

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Select all points
    idxgrid, Y = select_grid_points(data, 5)

    # Verify all points were selected
    assert len(idxgrid) == 5
    assert len(np.unique(idxgrid)) == 5

    # Verify that the selected points match the original data
    selected_points = set(tuple(point) for point in Y)
    original_points = set(tuple(point) for point in data)
    assert selected_points == original_points

    # Verify the order of selection follows the min-max algorithm
    # The first point is random, then subsequent points should maximize
    # the minimum distance to previously selected points

    # Create a copy of the data for tracking
    remaining_indices = set(range(5))

    # First point is random (with seed 42)
    first_idx = idxgrid[0]
    remaining_indices.remove(first_idx)

    # For each subsequent point, verify it maximizes the minimum distance
    for i in range(1, 5):
        current_idx = idxgrid[i]
        current_point = data[current_idx]

        # Calculate the minimum distance from this point to all previously selected points
        min_dist_selected = float("inf")
        for j in range(i):
            prev_idx = idxgrid[j]
            prev_point = data[prev_idx]
            dist = np.sum((current_point - prev_point) ** 2)  # Squared Euclidean
            min_dist_selected = min(min_dist_selected, dist)

        # For all other remaining points, calculate their minimum distances
        for other_idx in remaining_indices - {current_idx}:
            other_point = data[other_idx]
            min_dist_other = float("inf")
            for j in range(i):
                prev_idx = idxgrid[j]
                prev_point = data[prev_idx]
                dist = np.sum((other_point - prev_point) ** 2)  # Squared Euclidean
                min_dist_other = min(min_dist_other, dist)

            # The selected point should have a min distance >= any other remaining point
            assert min_dist_selected >= min_dist_other - 1e-10  # Allow for numerical precision

        remaining_indices.remove(current_idx)


def test_single_point_selection():
    """Test selecting a single point."""
    data = np.random.rand(10, 2)
    idxgrid, Y = select_grid_points(data, 1)

    assert idxgrid.shape == (1,)
    assert Y.shape == (1, 2)


def test_high_dimensional_data():
    """Test with high-dimensional data."""
    data = np.random.rand(20, 10)  # 20 points in 10D
    ngrid = 5
    idxgrid, Y = select_grid_points(data, ngrid)

    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, 10)


def test_reproducibility():
    """Test that results are reproducible with the same random seed."""
    data = np.random.rand(20, 3)

    # Set the same seed before each call
    np.random.seed(42)
    idxgrid1, Y1 = select_grid_points(data, 5)

    np.random.seed(42)
    idxgrid2, Y2 = select_grid_points(data, 5)

    # Results should be identical
    np.testing.assert_array_equal(idxgrid1, idxgrid2)
    np.testing.assert_array_equal(Y1, Y2)


# Test with different data types and structures
def test_integer_data():
    """Test with integer data."""
    # Convert integer data to float64
    data = np.random.randint(0, 100, size=(20, 3)).astype(np.float64)
    idxgrid, Y = select_grid_points(data, 5)

    assert idxgrid.shape == (5,)
    assert Y.shape == (5, 3)

    # Y should contain the exact same values as the original data points
    for i, idx in enumerate(idxgrid):
        np.testing.assert_array_equal(Y[i], data[idx])


def test_with_duplicate_points():
    """Test behavior with duplicate points in the dataset."""
    # Create data with some duplicate points, ensuring float64 type
    data = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],  # Duplicate
            [2.0, 2.0],
            [1.0, 1.0],  # Duplicate
        ],
        dtype=np.float64,
    )

    idxgrid, Y = select_grid_points(data, 3)

    # Should select 3 unique points (even though there are only 3 unique coordinates)
    assert len(np.unique(Y, axis=0)) == 3


# Additional tests for edge cases and specific behaviors


def test_empty_data():
    """Test with empty data."""
    data = np.array([]).reshape(0, 3)

    # This should raise a ValueError or similar
    with pytest.raises(Exception):
        select_grid_points(data, 5)


def test_one_dimensional_data():
    """Test with one-dimensional data."""
    data = np.random.rand(10, 1)  # 10 points in 1D
    ngrid = 5
    idxgrid, Y = select_grid_points(data, ngrid)

    assert idxgrid.shape == (ngrid,)
    assert Y.shape == (ngrid, 1)


def test_different_minkowski_exponents():
    """Test Minkowski distance with different exponents."""
    data = np.random.rand(20, 2)

    # Test with p=1 (Manhattan)
    p1 = np.array([[1.0]])
    idxgrid1, Y1 = select_grid_points(data, 5, metric="minkowski", inv_cov=p1)

    # Test with p=2 (Euclidean)
    p2 = np.array([[2.0]])
    idxgrid2, Y2 = select_grid_points(data, 5, metric="minkowski", inv_cov=p2)

    # Test with p=inf (approximated with a large value, should be similar to Chebyshev)
    p_inf = np.array([[1000.0]])
    idxgrid_inf, Y_inf = select_grid_points(data, 5, metric="minkowski", inv_cov=p_inf)

    # The selected points should be different for different exponents
    # This is a probabilistic test, so it might occasionally fail
    assert not np.array_equal(idxgrid1, idxgrid2) or not np.array_equal(idxgrid2, idxgrid_inf)


def test_mahalanobis_with_non_identity_matrix():
    """Test Mahalanobis distance with a non-identity covariance matrix."""
    np.random.seed(42)
    data = np.random.rand(20, 2)

    # Create a non-identity inverse covariance matrix
    inv_cov = np.array([[2.0, 0.5], [0.5, 1.0]])

    idxgrid, Y = select_grid_points(data, 5, metric="mahalanobis", inv_cov=inv_cov)

    # Basic shape checks
    assert idxgrid.shape == (5,)
    assert Y.shape == (5, 2)

    # Compare with Euclidean distance (should select different points)
    idxgrid_euclidean, Y_euclidean = select_grid_points(data, 5)

    # The selected points should be different due to the different distance metric
    # This is a probabilistic test, so it might occasionally fail
    assert not np.array_equal(idxgrid, idxgrid_euclidean)


def test_cosine_with_zero_vectors():
    """Test cosine distance with vectors containing zeros."""
    # Create data with some zero vectors, ensuring float64 type
    data = np.array(
        [
            [0.0, 0.0],  # Zero vector
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )

    idxgrid, Y = select_grid_points(data, 3, metric="cosine")

    # Basic shape checks
    assert idxgrid.shape == (3,)
    assert Y.shape == (3, 2)


def test_numerical_stability():
    """Test numerical stability with very small and very large values."""
    # Create data with very small and very large values
    data = np.array([[1e-10, 1e-10], [1e-5, 1e-5], [1.0, 1.0], [1e5, 1e5], [1e10, 1e10]])

    # Test with different metrics
    for metric in ["euclidean", "manhattan", "chebyshev", "cosine"]:
        idxgrid, Y = select_grid_points(data, 3, metric=metric)

        # Basic shape checks
        assert idxgrid.shape == (3,)
        assert Y.shape == (3, 2)


# def test_voronoi_basic(simple_2d_data):
#     X, Y, wj, idxgrid = simple_2d_data
#     iminij, ni, wi, ineigh = compute_voronoi(X, wj, Y, idxgrid, metric="euclidean")

#     assert iminij.shape == (X.shape[0],)
#     assert ni.shape == (Y.shape[0],)
#     assert wi.shape == (Y.shape[0],)
#     assert ineigh.shape == (Y.shape[0],)
#     assert np.sum(ni) == X.shape[0]
#     np.testing.assert_allclose(np.sum(wi), np.sum(wj))


# def test_voronoi_single_grid_point(random_2d_data):
#     X, wj = random_2d_data
#     Y = np.array([[0.5, 0.5]], dtype=np.float64)
#     idxgrid = np.array([0], dtype=np.int32)

#     iminij, ni, wi, ineigh = compute_voronoi(X, wj, Y, idxgrid, metric="euclidean")

#     assert np.all(iminij == 0)
#     assert ni[0] == X.shape[0]
#     assert np.isclose(wi[0], np.sum(wj))


# def test_voronoi_zero_weights(random_2d_data):
#     X, _ = random_2d_data
#     wj = np.zeros(X.shape[0], dtype=np.float64)
#     Y = np.array([[0.0, 0.0], [1.0, 1.0]])
#     idxgrid = np.array([0, 1], dtype=np.int32)

#     iminij, ni, wi, ineigh = compute_voronoi(X, wj, Y, idxgrid, metric="euclidean")

#     assert np.sum(ni) == X.shape[0]
#     assert np.allclose(wi, 0.0)
#     assert ineigh.shape == (Y.shape[0],)


# def test_voronoi_invalid_metric(simple_2d_data):
#     X, Y, wj, idxgrid = simple_2d_data

#     with pytest.raises(ValueError, match="Unsupported metric"):
#         compute_voronoi(X, wj, Y, idxgrid, metric="banana")
