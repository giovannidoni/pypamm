import numpy as np
import pytest

from pypamm import quick_shift, quick_shift_kde


@pytest.fixture
def random_data():
    """Generate random data for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(100, 2)  # 100 points in 2D


@pytest.fixture
def simple_data():
    """Simple dataset with known structure for predictable tests."""
    # Create a grid of points in 2D with a clear structure
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.flatten(), yy.flatten()])  # 25 points in 2D


def test_vanilla_quick_shift(random_data):
    """Test the vanilla QuickShift implementation."""
    # Use uniform probabilities
    prob = np.ones(len(random_data)) / len(random_data)

    # Run vanilla QuickShift
    labels, centers = quick_shift(random_data, prob, ngrid=10)

    # Check that labels and centers have the expected shapes
    assert labels.shape == (len(random_data),)
    assert centers.shape[0] > 0  # At least one cluster center

    # Check that all points have valid cluster assignments
    assert np.all(labels >= 0)
    assert np.all(labels < len(random_data))

    # Check that all cluster centers are valid indices
    assert np.all(centers >= 0)
    assert np.all(centers < len(random_data))


def test_kde_quick_shift(random_data):
    """Test the KDE-enhanced QuickShift implementation."""
    # Run KDE-enhanced QuickShift
    labels, centers = quick_shift_kde(random_data, bandwidth=0.1, ngrid=10)

    # Check that labels and centers have the expected shapes
    assert labels.shape == (len(random_data),)
    assert centers.shape[0] > 0  # At least one cluster center

    # Check that all points have valid cluster assignments
    assert np.all(labels >= 0)
    assert np.all(labels < len(random_data))

    # Check that all cluster centers are valid indices
    assert np.all(centers >= 0)
    assert np.all(centers < len(random_data))


def test_compare_implementations(simple_data):
    """Compare the results of both QuickShift implementations."""
    # Compute KDE probabilities
    from pypamm.density import compute_kde

    bandwidth = 0.1
    prob = compute_kde(simple_data, simple_data, bandwidth)

    # Run vanilla QuickShift with pre-computed probabilities
    labels_vanilla, centers_vanilla = quick_shift(simple_data, prob, ngrid=10)

    # Run KDE-enhanced QuickShift
    labels_kde, centers_kde = quick_shift_kde(simple_data, bandwidth=0.1, ngrid=10)

    # The results should be similar (not necessarily identical due to implementation details)
    # Check that the number of clusters is similar
    assert abs(len(np.unique(labels_vanilla)) - len(np.unique(labels_kde))) <= 2


def test_max_dist_parameter(random_data):
    """Test the max_dist parameter in both implementations."""
    # Run vanilla QuickShift with a small max_dist
    labels_small_dist, centers_small_dist = quick_shift(random_data, None, ngrid=10, max_dist=0.1)

    # Run vanilla QuickShift with a large max_dist
    labels_large_dist, centers_large_dist = quick_shift(random_data, None, ngrid=10, max_dist=2.0)

    # With a smaller max_dist, we expect more clusters
    assert len(np.unique(labels_small_dist)) >= len(np.unique(labels_large_dist))

    # Run KDE-enhanced QuickShift with different max_dist values
    labels_kde_small, centers_kde_small = quick_shift_kde(random_data, bandwidth=0.1, ngrid=10, max_dist=0.1)

    labels_kde_large, centers_kde_large = quick_shift_kde(random_data, bandwidth=0.1, ngrid=10, max_dist=2.0)

    # With a smaller max_dist, we expect more clusters
    assert len(np.unique(labels_kde_small)) >= len(np.unique(labels_kde_large))


def test_lambda_parameter(random_data):
    """Test the lambda_qs parameter in both implementations."""
    # Run vanilla QuickShift with different lambda values
    labels_small_lambda, centers_small_lambda = quick_shift(
        random_data,
        None,
        ngrid=10,
        metric="euclidean",  # Explicitly specify metric as a string
        lambda_qs=0.5,
    )

    labels_large_lambda, centers_large_lambda = quick_shift(
        random_data,
        None,
        ngrid=10,
        metric="euclidean",  # Explicitly specify metric as a string
        lambda_qs=2.0,
    )

    # Lambda affects the clustering, but the exact effect depends on the data
    # Just check that the function runs without errors
    assert labels_small_lambda.shape == (len(random_data),)
    assert labels_large_lambda.shape == (len(random_data),)

    # Run KDE-enhanced QuickShift with different lambda values
    labels_kde_small_lambda, centers_kde_small_lambda = quick_shift_kde(
        random_data,
        bandwidth=0.1,
        ngrid=10,
        metric="euclidean",  # Explicitly specify metric as a string
        lambda_qs=0.5,
    )

    labels_kde_large_lambda, centers_kde_large_lambda = quick_shift_kde(
        random_data,
        bandwidth=0.1,
        ngrid=10,
        metric="euclidean",  # Explicitly specify metric as a string
        lambda_qs=2.0,
    )

    # Check that the function runs without errors
    assert labels_kde_small_lambda.shape == (len(random_data),)
    assert labels_kde_large_lambda.shape == (len(random_data),)
