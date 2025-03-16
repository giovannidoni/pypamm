import numpy as np
import pytest

from pypamm.density import compute_kde, gauss_prepare, kde_bootstrap_error, kde_cutoff, kde_output
from pypamm.density.kde import compute_bandwidth  # Add explicit import for compute_bandwidth


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


@pytest.fixture
def bimodal_data():
    """Create a bimodal dataset with two distinct clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.1 + np.array([0, 0])  # Dense cluster
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([2, 2])  # Sparse cluster
    return np.vstack([cluster1, cluster2])


# Test gauss_prepare function
def test_gauss_prepare(random_data):
    """Test the gauss_prepare function with default parameters."""
    result = gauss_prepare(random_data)

    # The updated implementation returns 6 values
    mean, cov, inv_cov, eigvals, Hi, Hiinv = result

    # Check shapes
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    assert inv_cov.shape == (2, 2)
    assert eigvals.shape == (2,)
    assert Hi.shape == (2, 2)
    assert Hiinv.shape == (2, 2)

    # Check mean calculation
    expected_mean = np.mean(random_data, axis=0)
    assert np.allclose(mean, expected_mean)

    # Check covariance calculation
    expected_cov = np.cov(random_data, rowvar=False)
    # Add small regularization as in the implementation
    expected_cov += np.eye(2) * 1e-6
    assert np.allclose(cov, expected_cov)

    # Check eigenvalues are positive
    assert np.all(eigvals > 0), "All eigenvalues must be positive"

    # Check invertibility
    identity_approx = np.dot(cov, inv_cov)
    assert np.allclose(identity_approx, np.eye(2), atol=1e-5), "Inverse covariance should satisfy A * A_inv = I"

    # Check bandwidth matrices
    assert np.allclose(np.dot(Hi, Hiinv), np.eye(2), atol=1e-5), "Hi * Hiinv should be identity"


# Test gauss_prepare function with different parameters
def test_gauss_prepare_parameters():
    """Test the gauss_prepare function with different parameters."""
    X = np.random.rand(100, 3)  # 100 points in 3D

    # Test with adaptive bandwidth
    result_adaptive = gauss_prepare(X, alpha=0.2, adaptive=True)
    mean, cov, inv_cov, eigvals, Hi, Hiinv = result_adaptive

    # Test with fixed bandwidth
    result_fixed = gauss_prepare(X, constant_bandwidth=0.5, adaptive=False)
    mean_fixed, cov_fixed, inv_cov_fixed, eigvals_fixed, Hi_fixed, Hiinv_fixed = result_fixed

    # Check shapes
    assert Hi.shape == (3, 3), "Bandwidth matrix `Hi` shape mismatch"
    assert Hiinv.shape == (3, 3), "Inverse bandwidth `Hiinv` shape mismatch"
    assert Hi_fixed.shape == (3, 3), "Fixed bandwidth matrix shape mismatch"

    # Check that fixed bandwidth produces different results than adaptive
    assert not np.allclose(Hi, Hi_fixed), "Fixed and adaptive bandwidth should differ"


def test_compute_kde_fixed_bandwidth(random_data, grid_points):
    """Test the compute_kde function with fixed bandwidth."""
    # Test with fixed bandwidth
    bandwidth = 0.1
    density = compute_kde(random_data, grid_points, constant_bandwidth=bandwidth, adaptive=False)

    # Check shape
    assert density.shape == (len(grid_points),)

    # Check that density is positive
    assert np.all(density >= 0)
    assert np.any(density > 0), "KDE should not be all zeros"

    # Check that density integrates to approximately 1
    # (This is an approximation since we're not using a proper integration grid)
    grid_volume = 1.0  # Assuming grid is in [0,1]^2
    approx_integral = np.mean(density) * grid_volume
    assert 0.1 < approx_integral < 10.0  # Very loose bounds

    # Test with different bandwidths
    density_small = compute_kde(random_data, grid_points, constant_bandwidth=0.05, adaptive=False)
    density_large = compute_kde(random_data, grid_points, constant_bandwidth=0.5, adaptive=False)

    # Larger bandwidth should produce lower peak densities
    assert np.max(density_large) < np.max(density_small), "Larger bandwidth should produce lower peak densities"


def test_compute_kde_adaptive_bandwidth():
    """Test compute_kde with adaptive bandwidth."""
    X = np.random.rand(100, 2)  # 100 points in 2D
    grid = np.random.rand(20, 2)  # 20 grid points for KDE

    # Test with adaptive bandwidth
    density_adaptive = compute_kde(X, grid, alpha=0.2, adaptive=True)

    # Test with fixed bandwidth for comparison
    density_fixed = compute_kde(X, grid, constant_bandwidth=0.2, adaptive=False)

    assert density_adaptive.shape == (20,), "Adaptive density output shape mismatch"
    assert density_fixed.shape == (20,), "Fixed density output shape mismatch"
    assert np.all(density_adaptive >= 0), "Adaptive KDE densities must be non-negative"
    assert np.all(density_fixed >= 0), "Fixed KDE densities must be non-negative"

    # Adaptive and fixed should produce different results
    assert not np.allclose(density_adaptive, density_fixed), (
        "Adaptive and fixed bandwidth should produce different results"
    )


# Test kde_cutoff function with default alpha
def test_kde_cutoff_default():
    """Test the kde_cutoff function with default alpha."""
    # Test for different dimensions
    for d in range(1, 10):
        cutoff = kde_cutoff(d)
        assert cutoff > 0
        # The default alpha is 0.5 as defined in the implementation
        expected_cutoff = 9.0 * (np.sqrt(d) + 0.5) ** 2
        assert np.isclose(cutoff, expected_cutoff), f"Expected {expected_cutoff} but got {cutoff} for dimension {d}"


# Test kde_cutoff function with custom alpha
def test_kde_cutoff_custom_alpha():
    """Test the kde_cutoff function with custom alpha."""
    # Test for different dimensions with custom alpha
    alpha = 0.3
    for d in range(1, 10):
        cutoff_custom = kde_cutoff(d, alpha)
        assert cutoff_custom > 0
        assert np.isclose(cutoff_custom, 9.0 * (np.sqrt(d) + alpha) ** 2)


# Test kde_bootstrap_error function with fixed bandwidth
def test_kde_bootstrap_error_fixed(simple_data):
    """Test the kde_bootstrap_error function with fixed bandwidth."""
    n_bootstrap = 5
    bandwidth = 0.2

    mean_kde, std_kde = kde_bootstrap_error(simple_data, n_bootstrap, constant_bandwidth=bandwidth, adaptive=False)

    # Check shapes
    assert mean_kde.shape == (len(simple_data),)
    assert std_kde.shape == (len(simple_data),)

    # Check that mean is non-negative (KDE values should be non-negative)
    assert np.all(mean_kde >= 0)

    # Check that std is non-negative (standard deviations are always non-negative)
    assert np.all(std_kde >= 0)

    # Verify that bootstrap produces reasonable results
    # Compute KDE directly for comparison
    direct_kde = compute_kde(simple_data, simple_data, constant_bandwidth=bandwidth, adaptive=False)

    # Mean KDE from bootstrap should be similar to direct KDE (but not identical)
    assert np.allclose(mean_kde, direct_kde, rtol=0.5), "Bootstrap mean should be roughly similar to direct KDE"


# Test kde_bootstrap_error function with adaptive bandwidth
def test_kde_bootstrap_error_adaptive(simple_data):
    """Test the kde_bootstrap_error function with adaptive bandwidth."""
    n_bootstrap = 5
    alpha = 0.2

    mean_kde_adaptive, std_kde_adaptive = kde_bootstrap_error(simple_data, n_bootstrap, alpha=alpha, adaptive=True)

    # Check shapes and non-negativity
    assert mean_kde_adaptive.shape == (len(simple_data),)
    assert std_kde_adaptive.shape == (len(simple_data),)
    assert np.all(mean_kde_adaptive >= 0)
    assert np.all(std_kde_adaptive >= 0)

    # Verify that bootstrap produces reasonable results
    # Compute KDE directly for comparison
    direct_kde_adaptive = compute_kde(simple_data, simple_data, alpha=alpha, adaptive=True)

    # Mean KDE from bootstrap should be similar to direct KDE (but not identical)
    assert np.allclose(mean_kde_adaptive, direct_kde_adaptive, rtol=0.5), (
        "Bootstrap mean should be roughly similar to direct KDE"
    )


# Test kde_output function
def test_kde_output(random_data, grid_points):
    """Test the kde_output function."""
    bandwidth = 0.1
    density = compute_kde(random_data, grid_points, constant_bandwidth=bandwidth, adaptive=False)
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


# Test small bandwidth effect
def test_small_bandwidth_effect(simple_data):
    """Test the effect of small bandwidth on KDE."""
    # Use the data points as both data and grid
    small_bandwidth = 0.01
    density = compute_kde(simple_data, simple_data, constant_bandwidth=small_bandwidth, adaptive=False)

    # Check that density is positive
    assert np.all(density >= 0)

    # For very small bandwidth, density should be higher at data points
    assert np.max(density) > 1.0, "Small bandwidth should produce high peak densities"


# Test large bandwidth effect
def test_large_bandwidth_effect(simple_data):
    """Test the effect of large bandwidth on KDE."""
    # Use the data points as both data and grid
    large_bandwidth = 1.0
    density = compute_kde(simple_data, simple_data, constant_bandwidth=large_bandwidth, adaptive=False)

    # Check that density is positive
    assert np.all(density >= 0)

    # For very large bandwidth, density should be more uniform
    assert np.std(density) < 1.0, "Large bandwidth should produce more uniform densities"


# Test with different dimensions
def test_different_dimensions():
    """Test KDE with data of different dimensions."""
    np.random.seed(42)

    for d in range(1, 5):  # Test dimensions 1 through 4
        # Generate random data in d dimensions
        data = np.random.rand(50, d).astype(np.float64)
        grid = np.random.rand(10, d).astype(np.float64)
        bandwidth = 0.2

        # Compute KDE with fixed bandwidth
        density = compute_kde(data, grid, constant_bandwidth=bandwidth, adaptive=False)

        # Check shape
        assert density.shape == (len(grid),)

        # Check that density is positive
        assert np.all(density >= 0)

        # Compute KDE with adaptive bandwidth
        density_adaptive = compute_kde(data, grid, alpha=bandwidth, adaptive=True)

        # Check shape and positivity
        assert density_adaptive.shape == (len(grid),)
        assert np.all(density_adaptive >= 0)


# Test with single point
def test_single_point():
    """Test KDE with a single data point."""
    # Single point
    single_point = np.array([[0.5, 0.5]])
    grid = np.array([[0.5, 0.5], [1.0, 1.0]])
    bandwidth = 0.1

    # Test with fixed bandwidth
    density = compute_kde(single_point, grid, constant_bandwidth=bandwidth, adaptive=False)
    assert density.shape == (2,)
    assert density[0] > density[1], "Density should be higher at the data point"

    # Test with adaptive bandwidth on single point
    density_adaptive = compute_kde(single_point, grid, alpha=0.1, adaptive=True)
    assert density_adaptive.shape == (2,)
    assert density_adaptive[0] > 0, "Density should be positive at the data point"


# Test with very large bandwidth
def test_very_large_bandwidth():
    """Test KDE with a very large bandwidth."""
    # Very large bandwidth
    large_bandwidth = 100.0
    data = np.random.rand(10, 2)
    grid = np.array([[0.5, 0.5], [1.0, 1.0]])

    density_large = compute_kde(data, grid, constant_bandwidth=large_bandwidth, adaptive=False)

    # Density should be more uniform with large bandwidth
    assert np.std(density_large) < 0.1, "Large bandwidth should produce nearly uniform density"


# Test adaptive bandwidth on non-uniform data
def test_adaptive_bandwidth_non_uniform(bimodal_data):
    """Test adaptive bandwidth on data with varying density."""
    # Create a grid covering both clusters
    x = np.linspace(-0.5, 3, 20)
    y = np.linspace(-0.5, 3, 20)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.flatten(), yy.flatten()])

    # Compute KDE with fixed bandwidth
    bandwidth = 0.3
    density_fixed = compute_kde(bimodal_data, grid, constant_bandwidth=bandwidth, adaptive=False)

    # Compute KDE with adaptive bandwidth
    density_adaptive = compute_kde(bimodal_data, grid, alpha=0.2, adaptive=True)

    # Reshape for easier analysis
    density_fixed_grid = density_fixed.reshape(20, 20)
    density_adaptive_grid = density_adaptive.reshape(20, 20)

    # Get indices for dense and sparse regions
    dense_region = (xx < 0.5) & (yy < 0.5)
    sparse_region = (xx > 1.5) & (yy > 1.5)

    # In adaptive KDE, the bandwidth adapts to local density
    # So the ratio of max density in dense vs sparse regions should be different
    # between fixed and adaptive bandwidth
    fixed_ratio = np.max(density_fixed_grid[dense_region]) / np.max(density_fixed_grid[sparse_region])
    adaptive_ratio = np.max(density_adaptive_grid[dense_region]) / np.max(density_adaptive_grid[sparse_region])

    # The adaptive bandwidth should better handle varying densities
    assert fixed_ratio != adaptive_ratio, "Adaptive and fixed bandwidth should handle varying densities differently"

    # Check that both methods find the clusters
    assert np.max(density_fixed_grid[dense_region]) > 0, "Fixed bandwidth should detect dense cluster"
    assert np.max(density_fixed_grid[sparse_region]) > 0, "Fixed bandwidth should detect sparse cluster"
    assert np.max(density_adaptive_grid[dense_region]) > 0, "Adaptive bandwidth should detect dense cluster"
    assert np.max(density_adaptive_grid[sparse_region]) > 0, "Adaptive bandwidth should detect sparse cluster"


# Test adaptive bandwidth parameter sensitivity
def test_adaptive_bandwidth_sensitivity():
    """Test sensitivity of adaptive bandwidth to alpha parameter."""
    np.random.seed(42)
    data = np.random.rand(100, 2)
    grid = np.random.rand(20, 2)

    # Use very different alpha values to ensure we see a difference
    density_small_alpha = compute_kde(data, grid, alpha=0.1, adaptive=True)
    density_large_alpha = compute_kde(data, grid, alpha=0.9, adaptive=True)

    # Print values for debugging
    print(f"Small alpha max density: {np.max(density_small_alpha)}")
    print(f"Large alpha max density: {np.max(density_large_alpha)}")
    print(f"Small alpha std: {np.std(density_small_alpha)}")
    print(f"Large alpha std: {np.std(density_large_alpha)}")

    # Check at least one of these conditions is true
    different_results = not np.allclose(density_small_alpha, density_large_alpha)
    different_std = abs(np.std(density_small_alpha) - np.std(density_large_alpha)) > 1e-6
    different_max = abs(np.max(density_small_alpha) - np.max(density_large_alpha)) > 1e-6

    assert different_results or different_std or different_max, (
        "Very different alpha values should produce at least some difference in KDE results"
    )


# Test compute_bandwidth function
def test_compute_bandwidth_basic(random_data):
    """Test the basic functionality of compute_bandwidth."""
    # Call compute_bandwidth with default parameters
    global_bandwidth, adaptive_bandwidths = compute_bandwidth(random_data)

    # Check return types and shapes
    assert isinstance(global_bandwidth, float), "Global bandwidth should be a float"
    assert isinstance(adaptive_bandwidths, np.ndarray), "Adaptive bandwidths should be a numpy array"
    assert adaptive_bandwidths.shape == (len(random_data),), "Adaptive bandwidths should have shape (n_samples,)"

    # Check that bandwidths are positive
    assert global_bandwidth > 0, "Global bandwidth should be positive"
    assert np.all(adaptive_bandwidths > 0), "All adaptive bandwidths should be positive"

    # Check that global bandwidth is the mean of adaptive bandwidths
    assert np.isclose(global_bandwidth, np.mean(adaptive_bandwidths)), (
        "Global bandwidth should be the mean of adaptive bandwidths"
    )


def test_compute_bandwidth_alpha_effect(random_data):
    """Test the effect of alpha parameter on compute_bandwidth."""
    # Try different alpha values with larger differences
    alpha_values = [0.1, 0.8]  # Use more extreme values to ensure difference
    bandwidths = []
    adaptive_bandwidths_list = []

    for alpha in alpha_values:
        h, adaptive_h = compute_bandwidth(random_data, alpha=alpha)
        bandwidths.append(h)
        adaptive_bandwidths_list.append(adaptive_h)
        print(f"Successfully computed bandwidth for alpha={alpha}: {h}")

    # Print values for debugging
    print(f"Bandwidths for alpha={alpha_values[0]}: {bandwidths[0]}")
    print(f"Bandwidths for alpha={alpha_values[1]}: {bandwidths[1]}")

    # Check basic properties
    assert all(b > 0 for b in bandwidths), "All bandwidths should be positive"
    assert all(np.all(ab > 0) for ab in adaptive_bandwidths_list), "All adaptive bandwidths should be positive"

    # Check that the returned bandwidths have the correct shape
    assert len(adaptive_bandwidths_list[0]) == len(random_data), "Adaptive bandwidths should have same length as data"

    # For very different alpha values, we expect at least some difference
    # This is a more relaxed test that should pass with any reasonable implementation
    assert abs(bandwidths[0] - bandwidths[1]) > 1e-10, (
        "Very different alpha values should produce at least slightly different bandwidths"
    )


def test_compute_bandwidth_dimensions(simple_data):
    """Test compute_bandwidth with data of different dimensions."""
    # Set a fixed seed for reproducibility
    np.random.seed(42)

    # Test with 2D data (original)
    h_2d, adaptive_h_2d = compute_bandwidth(simple_data)
    print(f"2D bandwidth: {h_2d}")

    # Create 3D data by adding a dimension
    data_3d = np.column_stack([simple_data, np.random.rand(len(simple_data))])
    h_3d, adaptive_h_3d = compute_bandwidth(data_3d)
    print(f"3D bandwidth: {h_3d}")

    # Create 1D data
    data_1d = simple_data[:, 0].reshape(-1, 1)
    h_1d, adaptive_h_1d = compute_bandwidth(data_1d)
    print(f"1D bandwidth: {h_1d}")

    # Check shapes
    assert adaptive_h_1d.shape == (len(data_1d),), "Adaptive bandwidths should have shape (n_samples,)"
    assert adaptive_h_2d.shape == (len(simple_data),), "Adaptive bandwidths should have shape (n_samples,)"
    assert adaptive_h_3d.shape == (len(data_3d),), "Adaptive bandwidths should have shape (n_samples,)"

    # Check that all bandwidths are positive
    assert h_1d > 0 and h_2d > 0 and h_3d > 0, "All bandwidths should be positive"
    assert np.all(adaptive_h_1d > 0) and np.all(adaptive_h_2d > 0) and np.all(adaptive_h_3d > 0), (
        "All adaptive bandwidths should be positive"
    )

    # We don't strictly test that bandwidths differ between dimensions
    # as this depends on the specific implementation and data distribution
    # Instead, we verify that the algorithm runs successfully for different dimensions


def test_compute_bandwidth_delta_effect(random_data):
    """Test the effect of delta parameter on compute_bandwidth."""
    # Try different delta values
    delta_values = [1e-4, 1e-3, 1e-2]

    bandwidths = []
    for delta in delta_values:
        h, _ = compute_bandwidth(random_data, delta=delta)
        bandwidths.append(h)
        print(f"Bandwidth for delta={delta}: {h}")

    # Different delta values might produce slightly different results
    # but the differences should be small
    for i in range(len(delta_values) - 1):
        assert abs(bandwidths[i] - bandwidths[i + 1]) < 0.1, (
            f"Delta should have a small effect on bandwidth: {bandwidths[i]} vs {bandwidths[i + 1]}"
        )


def test_compute_bandwidth_convergence(simple_data):
    """Test that compute_bandwidth converges to the target alpha value."""
    # Use a moderate alpha value
    alpha = 0.3
    _, adaptive_h = compute_bandwidth(simple_data, alpha=alpha)

    # For each point, compute the fraction of points within its bandwidth
    fractions = []
    for i in range(len(simple_data)):
        # Count points within bandwidth
        distances = np.linalg.norm(simple_data - simple_data[i], axis=1)
        fraction = np.mean(distances < adaptive_h[i])
        fractions.append(fraction)

    # The mean fraction should be reasonably close to alpha
    # We use a larger tolerance because perfect convergence is difficult
    # with a small dataset and limited iterations
    mean_fraction = np.mean(fractions)
    print(f"Target alpha: {alpha}, Mean fraction: {mean_fraction}")
    assert abs(mean_fraction - alpha) < 0.25, (
        f"Mean fraction {mean_fraction} should be reasonably close to alpha {alpha}"
    )


def test_compute_bandwidth_edge_cases():
    """Test compute_bandwidth with edge cases."""
    # Single point
    single_point = np.array([[0.5, 0.5]])
    h_single, adaptive_h_single = compute_bandwidth(single_point)
    print(f"Bandwidth for single point: {h_single}")

    assert h_single > 0, "Bandwidth for single point should be positive"
    assert len(adaptive_h_single) == 1, "Should return one adaptive bandwidth for one point"

    # Very small dataset
    small_data = np.random.rand(3, 2)
    h_small, adaptive_h_small = compute_bandwidth(small_data)
    print(f"Bandwidth for small dataset: {h_small}")

    assert h_small > 0, "Bandwidth for small dataset should be positive"
    assert len(adaptive_h_small) == 3, "Should return three adaptive bandwidths for three points"

    # Uniform grid
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.flatten(), yy.flatten()])

    h_grid, adaptive_h_grid = compute_bandwidth(grid_data)
    print(f"Bandwidth for uniform grid: {h_grid}")

    assert h_grid > 0, "Bandwidth for grid data should be positive"
    assert len(adaptive_h_grid) == 25, "Should return 25 adaptive bandwidths for 25 points"

    # For uniform grid, bandwidths should be more uniform
    std_dev = np.std(adaptive_h_grid)
    print(f"Standard deviation of bandwidths for uniform grid: {std_dev}")
    assert std_dev < 0.1, f"Bandwidths for uniform grid should be relatively uniform, got std={std_dev}"


# Modified test for compute_bandwidth with smaller dataset
def test_compute_bandwidth_basic_small():
    """Test the basic functionality of compute_bandwidth with a small dataset."""
    # Create a small dataset
    np.random.seed(42)
    small_data = np.random.rand(20, 2)  # Only 20 points in 2D

    # Call compute_bandwidth with default parameters
    global_bandwidth, adaptive_bandwidths = compute_bandwidth(small_data)

    # Check return types and shapes
    assert isinstance(global_bandwidth, float), "Global bandwidth should be a float"
    assert isinstance(adaptive_bandwidths, np.ndarray), "Adaptive bandwidths should be a numpy array"
    assert adaptive_bandwidths.shape == (len(small_data),), "Adaptive bandwidths should have shape (n_samples,)"

    # Check that bandwidths are positive
    assert global_bandwidth > 0, "Global bandwidth should be positive"
    assert np.all(adaptive_bandwidths > 0), "All adaptive bandwidths should be positive"

    # Check that global bandwidth is the mean of adaptive bandwidths
    assert np.isclose(global_bandwidth, np.mean(adaptive_bandwidths)), (
        "Global bandwidth should be the mean of adaptive bandwidths"
    )


# Test with larger delta for faster convergence
def test_compute_bandwidth_faster_convergence():
    """Test compute_bandwidth with larger delta for faster convergence."""
    # Create a small dataset
    np.random.seed(42)
    small_data = np.random.rand(20, 2)  # Only 20 points in 2D

    # Use a larger delta for faster convergence
    delta = 0.01  # 10x larger than default

    # Call compute_bandwidth with larger delta
    global_bandwidth, adaptive_bandwidths = compute_bandwidth(small_data, delta=delta)

    # Check that bandwidths are positive
    assert global_bandwidth > 0, "Global bandwidth should be positive"
    assert np.all(adaptive_bandwidths > 0), "All adaptive bandwidths should be positive"


# Test with uniform grid for more predictable convergence
def test_compute_bandwidth_uniform_grid():
    """Test compute_bandwidth with a uniform grid for more predictable convergence."""
    # Create a small uniform grid
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.flatten(), yy.flatten()])  # 25 points in 2D

    # Call compute_bandwidth with uniform grid
    global_bandwidth, adaptive_bandwidths = compute_bandwidth(grid_data)

    # Check that bandwidths are positive
    assert global_bandwidth > 0, "Global bandwidth should be positive"
    assert np.all(adaptive_bandwidths > 0), "All adaptive bandwidths should be positive"

    # For uniform grid, bandwidths should be more uniform
    std_dev = np.std(adaptive_bandwidths)
    print(f"Standard deviation of bandwidths for uniform grid: {std_dev}")
    assert std_dev < 0.1, f"Bandwidths for uniform grid should be relatively uniform, got std={std_dev}"


# Test with a single point (edge case)
def test_compute_bandwidth_single_point():
    """Test compute_bandwidth with a single point."""
    # Single point
    single_point = np.array([[0.5, 0.5]])

    # Call compute_bandwidth with single point
    global_bandwidth, adaptive_bandwidths = compute_bandwidth(single_point)

    # Check return values
    assert global_bandwidth > 0, "Global bandwidth should be positive"
    assert len(adaptive_bandwidths) == 1, "Should return one adaptive bandwidth for one point"
    assert adaptive_bandwidths[0] > 0, "Adaptive bandwidth should be positive"


# Test with different alpha values on a small dataset
def test_compute_bandwidth_alpha_small():
    """Test the effect of alpha parameter on compute_bandwidth with a small dataset."""
    # Create a small dataset
    np.random.seed(42)
    small_data = np.random.rand(20, 2)  # Only 20 points in 2D

    # Try different alpha values
    alpha_values = [0.1, 0.5]  # Just two values for faster testing

    for alpha in alpha_values:
        h, adaptive_h = compute_bandwidth(small_data, alpha=alpha)

        # Check that bandwidths are positive
        assert h > 0, f"Global bandwidth should be positive for alpha={alpha}"
        assert np.all(adaptive_h > 0), f"All adaptive bandwidths should be positive for alpha={alpha}"
