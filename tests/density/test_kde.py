import numpy as np
import pytest
from pypamm.density import (
    gauss_prepare,
    compute_kde,
    kde_cutoff,
    kde_bootstrap_error,
    kde_output
)

# Fixtures for test data
@pytest.fixture
def random_data():
    """Generate random data for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(100, 2)  # 100 points in 2D

@pytest.fixture
def grid_points():
    """Generate grid points for testing."""
    np.random.seed(43)  # Different seed
    return np.random.rand(20, 2)  # 20 grid points in 2D

@pytest.fixture
def simple_data():
    """Simple dataset with known structure for predictable tests."""
    # Create a grid of points in 2D with a clear structure
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.flatten(), yy.flatten()])  # 25 points in 2D

# Test gauss_prepare function
def test_gauss_prepare(random_data):
    """Test the gauss_prepare function."""
    mean, cov = gauss_prepare(random_data)
    
    # Check shapes
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    
    # Check mean calculation
    expected_mean = np.mean(random_data, axis=0)
    assert np.allclose(mean, expected_mean)
    
    # Check covariance calculation
    expected_cov = np.cov(random_data, rowvar=False)
    assert np.allclose(cov, expected_cov)

# Test compute_kde function
def test_compute_kde(random_data, grid_points):
    """Test the compute_kde function."""
    bandwidth = 0.1
    density = compute_kde(random_data, grid_points, bandwidth)
    
    # Check shape
    assert density.shape == (len(grid_points),)
    
    # Check that density is positive
    assert np.all(density >= 0)
    
    # Check that density integrates to approximately 1
    # (This is an approximation since we're not using a proper integration grid)
    grid_volume = 1.0  # Assuming grid is in [0,1]^2
    approx_integral = np.mean(density) * grid_volume
    assert 0.1 < approx_integral < 10.0  # Very loose bounds

# Test kde_cutoff function
def test_kde_cutoff():
    """Test the kde_cutoff function."""
    # Test for different dimensions
    for d in range(1, 10):
        cutoff = kde_cutoff(d)
        assert cutoff > 0
        assert cutoff == 9.0 * (np.sqrt(d) + 1.0) ** 2

# Test kde_bootstrap_error function
def test_kde_bootstrap_error(simple_data):
    """Test the kde_bootstrap_error function."""
    n_bootstrap = 5
    bandwidth = 0.2
    
    mean_kde, std_kde = kde_bootstrap_error(simple_data, n_bootstrap, bandwidth)
    
    # Check shapes
    assert mean_kde.shape == (len(simple_data),)
    assert std_kde.shape == (len(simple_data),)
    
    # Check that mean is positive
    assert np.all(mean_kde >= 0)
    
    # Check that std is non-negative
    assert np.all(std_kde >= 0)

# Test kde_output function
def test_kde_output(random_data, grid_points):
    """Test the kde_output function."""
    bandwidth = 0.1
    density = compute_kde(random_data, grid_points, bandwidth)
    std_kde = np.random.rand(len(grid_points)) * 0.01  # Small random errors
    
    prb, aer, rer = kde_output(density, std_kde)
    
    # Check shapes
    assert prb.shape == (len(grid_points),)
    assert aer.shape == (len(grid_points),)
    assert rer.shape == (len(grid_points),)
    
    # Check that prb is the same as density
    assert np.array_equal(prb, density)
    
    # Check that aer is the same as std_kde
    assert np.array_equal(aer, std_kde)
    
    # Check that rer is std_kde / density (with protection against division by zero)
    expected_rer = std_kde / (density + 1e-8)
    assert np.allclose(rer, expected_rer)

# Test with different bandwidths
@pytest.mark.parametrize("bandwidth", [0.01, 0.1, 1.0])
def test_bandwidth_effect(simple_data, bandwidth):
    """Test the effect of different bandwidths on KDE."""
    # Use the data points as both data and grid
    density = compute_kde(simple_data, simple_data, bandwidth)
    
    # Check that density is positive
    assert np.all(density >= 0)
    
    # For very small bandwidth, density should be higher at data points
    if bandwidth == 0.01:
        # Density should be higher at data points
        assert np.max(density) > 1.0
    
    # For very large bandwidth, density should be more uniform
    if bandwidth == 1.0:
        # Density should be more uniform
        assert np.std(density) < 1.0

# Test with different dimensions
def test_different_dimensions():
    """Test KDE with data of different dimensions."""
    np.random.seed(42)
    
    for d in range(1, 5):  # Test dimensions 1 through 4
        # Generate random data in d dimensions
        data = np.random.rand(50, d)
        grid = np.random.rand(10, d)
        bandwidth = 0.2
        
        # Compute KDE
        density = compute_kde(data, grid, bandwidth)
        
        # Check shape
        assert density.shape == (len(grid),)
        
        # Check that density is positive
        assert np.all(density >= 0)

# Test with edge cases
def test_edge_cases():
    """Test KDE with edge cases."""
    # Single point
    single_point = np.array([[0.5, 0.5]])
    grid = np.array([[0.5, 0.5], [1.0, 1.0]])
    bandwidth = 0.1
    
    density = compute_kde(single_point, grid, bandwidth)
    assert density.shape == (2,)
    assert density[0] > density[1]  # Density should be higher at the data point
    
    # Very large bandwidth
    large_bandwidth = 100.0
    data = np.random.rand(10, 2)
    density = compute_kde(data, grid, large_bandwidth)
    
    # With very large bandwidth, density should be nearly uniform
    assert np.std(density) < 0.01
    
    # Very small bandwidth
    small_bandwidth = 1e-5
    density = compute_kde(single_point, grid, small_bandwidth)
    
    # With very small bandwidth, density should be nearly zero except at data points
    assert density[0] > 0
    assert density[1] < 1e-10 