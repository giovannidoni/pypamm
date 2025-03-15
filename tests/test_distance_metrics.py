import numpy as np
import pytest
from pypamm.distance_metrics import get_distance_function

# Fixtures for common test data
@pytest.fixture
def random_vectors():
    """Generate random vectors for testing distance functions."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(10, 3)  # 10 vectors in 3D

@pytest.fixture
def simple_vectors():
    """Simple vectors with known distances for predictable tests."""
    return np.array([
        [0.0, 0.0, 0.0],  # Origin
        [1.0, 0.0, 0.0],  # Unit vector along x
        [0.0, 1.0, 0.0],  # Unit vector along y
        [0.0, 0.0, 1.0],  # Unit vector along z
        [1.0, 1.0, 1.0],  # (1,1,1) vector
    ], dtype=np.float64)

# Test get_distance_function utility
def test_get_distance_function_basic():
    """Test basic functionality of get_distance_function."""
    # Test with euclidean metric
    dist_func, inv_cov = get_distance_function("euclidean")
    assert dist_func is not None
    assert inv_cov.shape == (1, 1)
    
    # Test with manhattan metric
    dist_func, inv_cov = get_distance_function("manhattan")
    assert dist_func is not None
    assert inv_cov.shape == (1, 1)
    
    # Test with chebyshev metric
    dist_func, inv_cov = get_distance_function("chebyshev")
    assert dist_func is not None
    assert inv_cov.shape == (1, 1)
    
    # Test with cosine metric
    dist_func, inv_cov = get_distance_function("cosine")
    assert dist_func is not None
    assert inv_cov.shape == (1, 1)

def test_get_distance_function_with_parameters():
    """Test get_distance_function with parameters for Mahalanobis and Minkowski."""
    # Test with Mahalanobis metric
    inv_cov_mat = np.eye(3)  # 3x3 identity matrix
    dist_func, inv_cov = get_distance_function("mahalanobis", inv_cov_mat, 3)
    assert dist_func is not None
    assert inv_cov.shape == (3, 3)
    np.testing.assert_array_equal(inv_cov, inv_cov_mat)
    
    # Test with Minkowski metric
    p_param = np.array([[3.0]])  # p=3
    dist_func, inv_cov = get_distance_function("minkowski", p_param)
    assert dist_func is not None
    assert inv_cov.shape == (1, 1)
    assert inv_cov[0, 0] == 3.0

def test_get_distance_function_errors():
    """Test error handling in get_distance_function."""
    # Test with invalid metric
    with pytest.raises(ValueError, match="Unsupported metric"):
        get_distance_function("invalid_metric")
    
    # Test Mahalanobis without inv_cov
    with pytest.raises(ValueError, match="Must supply inv_cov"):
        get_distance_function("mahalanobis")
    
    # Test Mahalanobis with wrong inv_cov shape
    with pytest.raises(ValueError, match="inv_cov must be"):
        get_distance_function("mahalanobis", np.eye(2), 3)
    
    # Test Minkowski without parameter
    with pytest.raises(ValueError, match="Must supply a 1x1 array"):
        get_distance_function("minkowski")
    
    # Test Minkowski with wrong parameter shape
    with pytest.raises(ValueError, match="inv_cov must be a 1x1 array"):
        get_distance_function("minkowski", np.array([[1.0, 2.0]]))

# Test distance calculations
def test_euclidean_distance(simple_vectors):
    """Test Euclidean distance calculations."""
    dist_func, inv_cov = get_distance_function("euclidean")
    
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # Distance from origin to (1,0,0) should be 1
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)
    
    # Distance from origin to (1,1,1) should be 3 (squared Euclidean)
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 3.0)
    
    # Distance from (1,0,0) to (1,1,1) should be 2 (squared Euclidean)
    dist_b_c = dist_func(b_view, c_view, inv_cov_view)
    assert np.isclose(dist_b_c, 2.0)

def test_manhattan_distance(simple_vectors):
    """Test Manhattan distance calculations."""
    dist_func, inv_cov = get_distance_function("manhattan")
    
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # Distance from origin to (1,0,0) should be 1
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)
    
    # Distance from origin to (1,1,1) should be 3
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 3.0)
    
    # Distance from (1,0,0) to (1,1,1) should be 2
    dist_b_c = dist_func(b_view, c_view, inv_cov_view)
    assert np.isclose(dist_b_c, 2.0)

def test_chebyshev_distance(simple_vectors):
    """Test Chebyshev distance calculations."""
    dist_func, inv_cov = get_distance_function("chebyshev")
    
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # Distance from origin to (1,0,0) should be 1
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)
    
    # Distance from origin to (1,1,1) should be 1 (max of |1|, |1|, |1|)
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 1.0)
    
    # Distance from (1,0,0) to (1,1,1) should be 1 (max of |0|, |1|, |1|)
    dist_b_c = dist_func(b_view, c_view, inv_cov_view)
    assert np.isclose(dist_b_c, 1.0)

def test_cosine_distance(simple_vectors):
    """Test Cosine distance calculations."""
    dist_func, inv_cov = get_distance_function("cosine")
    
    # Calculate distances between vectors
    a = simple_vectors[1]  # (1,0,0)
    b = simple_vectors[2]  # (0,1,0)
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # Distance between orthogonal vectors should be 1
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)
    
    # Distance from (1,0,0) to (1,1,1) should be 1 - 1/sqrt(3) ≈ 0.4226
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 1.0 - 1.0/np.sqrt(3.0))

def test_mahalanobis_distance(simple_vectors):
    """Test Mahalanobis distance calculations."""
    # Use identity matrix for simplicity (reduces to squared Euclidean)
    inv_cov_mat = np.eye(3)
    dist_func, inv_cov = get_distance_function("mahalanobis", inv_cov_mat)
    
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    b = simple_vectors[1]  # (1,0,0)
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # With identity matrix, should be same as squared Euclidean
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    assert np.isclose(dist_a_b, 1.0)
    
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 3.0)

def test_minkowski_distance(simple_vectors):
    """Test Minkowski distance calculations."""
    # Test with p=1 (Manhattan)
    p1 = np.array([[1.0]])
    dist_func, inv_cov = get_distance_function("minkowski", p1)
    
    # Calculate distances between vectors
    a = simple_vectors[0]  # Origin
    c = simple_vectors[4]  # (1,1,1)
    
    # Convert to memory views for the Cython function
    a_view = np.ascontiguousarray(a)
    c_view = np.ascontiguousarray(c)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # With p=1, should be same as Manhattan
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, 3.0)
    
    # Test with p=2 (Euclidean, but not squared)
    p2 = np.array([[2.0]])
    dist_func, inv_cov = get_distance_function("minkowski", p2)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # With p=2, should be Euclidean (not squared)
    dist_a_c = dist_func(a_view, c_view, inv_cov_view)
    assert np.isclose(dist_a_c, np.sqrt(3.0))

# Test edge cases
def test_zero_vectors():
    """Test distance calculations with zero vectors."""
    # Create zero vectors
    zero_vec = np.zeros(3)
    zero_view = np.ascontiguousarray(zero_vec)
    
    # Test with cosine distance (special handling for zero vectors)
    dist_func, inv_cov = get_distance_function("cosine")
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    # Distance between zero vectors should be 1 (by definition in our implementation)
    dist = dist_func(zero_view, zero_view, inv_cov_view)
    assert np.isclose(dist, 1.0)

def test_identical_vectors(random_vectors):
    """Test distance calculations with identical vectors."""
    # Get a random vector
    vec = random_vectors[0]
    vec_view = np.ascontiguousarray(vec)
    
    # Test with different metrics
    for metric in ["euclidean", "manhattan", "chebyshev", "cosine"]:
        dist_func, inv_cov = get_distance_function(metric)
        inv_cov_view = np.ascontiguousarray(inv_cov)
        
        # Distance between identical vectors should be 0 (except for cosine)
        dist = dist_func(vec_view, vec_view, inv_cov_view)
        if metric == "cosine":
            assert np.isclose(dist, 0.0)  # 1 - cos(0) = 0
        else:
            assert np.isclose(dist, 0.0)

def test_symmetry(random_vectors):
    """Test that distances are symmetric (dist(a,b) = dist(b,a))."""
    # Get two random vectors
    a = random_vectors[0]
    b = random_vectors[1]
    a_view = np.ascontiguousarray(a)
    b_view = np.ascontiguousarray(b)
    
    # Test with different metrics
    for metric in ["euclidean", "manhattan", "chebyshev", "cosine"]:
        dist_func, inv_cov = get_distance_function(metric)
        inv_cov_view = np.ascontiguousarray(inv_cov)
        
        # Distance should be symmetric
        dist_a_b = dist_func(a_view, b_view, inv_cov_view)
        dist_b_a = dist_func(b_view, a_view, inv_cov_view)
        assert np.isclose(dist_a_b, dist_b_a)
    
    # Test with Mahalanobis (symmetric if inv_cov is symmetric)
    inv_cov_mat = np.eye(3)
    dist_func, inv_cov = get_distance_function("mahalanobis", inv_cov_mat)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    dist_b_a = dist_func(b_view, a_view, inv_cov_view)
    assert np.isclose(dist_a_b, dist_b_a)
    
    # Test with Minkowski
    p = np.array([[2.0]])
    dist_func, inv_cov = get_distance_function("minkowski", p)
    inv_cov_view = np.ascontiguousarray(inv_cov)
    
    dist_a_b = dist_func(a_view, b_view, inv_cov_view)
    dist_b_a = dist_func(b_view, a_view, inv_cov_view)
    assert np.isclose(dist_a_b, dist_b_a) 