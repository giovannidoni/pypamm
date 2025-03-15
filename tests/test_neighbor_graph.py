import numpy as np
import pytest
from pypamm.neighbor_graph_wrapper import build_neighbor_graph
from pypamm.distance_metrics import get_distance_function

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
    data = np.array([
        [0.0, 0.0],  # Origin
        [1.0, 0.0],  # Right
        [0.0, 1.0],  # Top
        [1.0, 1.0],  # Top-right
        [0.5, 0.5]   # Center
    ], dtype=np.float64)
    return data

# Test basic functionality
def test_build_neighbor_graph_basic(random_data):
    """Test basic functionality of build_neighbor_graph with default parameters."""
    k = 3
    adjacency_list = build_neighbor_graph(random_data, k)
    
    # Check that the adjacency list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k
        
    # Check that each neighbor entry is a tuple of (index, distance)
    for neighbors in adjacency_list:
        for neighbor in neighbors:
            assert isinstance(neighbor, tuple)
            assert len(neighbor) == 2
            assert isinstance(neighbor[0], (int, np.integer))
            assert isinstance(neighbor[1], (float, np.floating))

# Test with different metrics
@pytest.mark.parametrize("metric", [
    "euclidean", "manhattan", "chebyshev", "cosine"
])
def test_different_metrics(random_data, metric):
    """Test build_neighbor_graph with different distance metrics."""
    k = 3
    adjacency_list = build_neighbor_graph(random_data, k, metric=metric)
    
    # Check that the adjacency list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

# Test with Mahalanobis distance
def test_mahalanobis_distance(random_data):
    """Test build_neighbor_graph with Mahalanobis distance."""
    k = 3
    D = random_data.shape[1]
    
    # Create an identity matrix for simplicity (reduces to Euclidean)
    inv_cov = np.eye(D)
    
    adjacency_list = build_neighbor_graph(
        random_data, k, inv_cov=inv_cov, metric="mahalanobis"
    )
    
    # Check that the list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

# Test with Minkowski distance
def test_minkowski_distance(random_data):
    """Test build_neighbor_graph with Minkowski distance."""
    k = 3
    
    # Create parameter for p=2 (Euclidean)
    p = np.array([[2.0]])
    
    adjacency_list = build_neighbor_graph(
        random_data, k, inv_cov=p, metric="minkowski"
    )
    
    # Check that the list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

# Test error handling
def test_invalid_metric(random_data):
    """Test that an invalid metric raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_neighbor_graph(random_data, 3, metric="invalid_metric")

def test_mahalanobis_without_inv_cov(random_data):
    """Test that Mahalanobis without inv_cov raises a ValueError."""
    with pytest.raises(ValueError, match="Must supply inv_cov"):
        build_neighbor_graph(random_data, 3, metric="mahalanobis")

def test_mahalanobis_wrong_shape(random_data):
    """Test that Mahalanobis with wrong inv_cov shape raises a ValueError."""
    D = random_data.shape[1]
    with pytest.raises(ValueError, match="inv_cov must be"):
        build_neighbor_graph(random_data, 3, metric="mahalanobis", inv_cov=np.eye(D+1))

def test_minkowski_without_param(random_data):
    """Test that Minkowski without parameter raises a ValueError."""
    with pytest.raises(ValueError, match="Must supply a 1x1 array"):
        build_neighbor_graph(random_data, 3, metric="minkowski")

def test_minkowski_wrong_shape(random_data):
    """Test that Minkowski with wrong parameter shape raises a ValueError."""
    with pytest.raises(ValueError, match="inv_cov must be a 1x1 array"):
        build_neighbor_graph(random_data, 3, metric="minkowski", inv_cov=np.array([[1.0, 2.0]]))

# Test with different search methods
def test_kd_tree_method(random_data):
    """Test build_neighbor_graph with KD-tree search method."""
    k = 3
    
    # Use KD-tree method with Euclidean distance
    adjacency_list = build_neighbor_graph(random_data, k, method="kd_tree", metric="euclidean")
    
    # Check that the adjacency list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

def test_brute_force_method(random_data):
    """Test build_neighbor_graph with brute force search method."""
    k = 3
    
    # Use brute force method
    adjacency_list = build_neighbor_graph(random_data, k, method="brute_force")
    
    # Check that the adjacency list has the correct length
    assert len(adjacency_list) == len(random_data)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

# Test correctness of neighbor search
def test_neighbor_search_correctness(simple_data):
    """Test that the neighbor search returns the correct neighbors."""
    k = 2  # Find 2 nearest neighbors
    
    # For this simple dataset, we know the exact neighbors for each point
    # Point 0 (0,0): Nearest are point 4 (0.5,0.5) and point 1 (1,0)
    # Point 1 (1,0): Nearest are point 4 (0.5,0.5) and point 0 (0,0)
    # Point 2 (0,1): Nearest are point 4 (0.5,0.5) and point 0 (0,0)
    # Point 3 (1,1): Nearest are point 4 (0.5,0.5) and point 1 (1,0) or point 2 (0,1)
    # Point 4 (0.5,0.5): Nearest are any of the other points (equidistant)
    
    adjacency_list = build_neighbor_graph(simple_data, k)
    
    # Check point 0 (0,0)
    neighbors_0 = [n[0] for n in adjacency_list[0]]  # Get indices only
    assert 4 in neighbors_0  # Point 4 (0.5,0.5) should be a neighbor
    
    # Check point 1 (1,0)
    neighbors_1 = [n[0] for n in adjacency_list[1]]
    assert 4 in neighbors_1  # Point 4 (0.5,0.5) should be a neighbor
    
    # Check point 2 (0,1)
    neighbors_2 = [n[0] for n in adjacency_list[2]]
    assert 4 in neighbors_2  # Point 4 (0.5,0.5) should be a neighbor
    
    # Check point 3 (1,1)
    neighbors_3 = [n[0] for n in adjacency_list[3]]
    assert 4 in neighbors_3  # Point 4 (0.5,0.5) should be a neighbor
    
    # Check point 4 (0.5,0.5) - should have 2 neighbors from the other 4 points
    neighbors_4 = [n[0] for n in adjacency_list[4]]
    assert len(set(neighbors_4).intersection({0, 1, 2, 3})) == 2

# Test symmetry of distances
def test_distance_symmetry(random_data):
    """Test that distances are symmetric (dist(a,b) = dist(b,a))."""
    k = len(random_data) - 1  # Find all neighbors
    
    adjacency_list = build_neighbor_graph(random_data, k)
    
    # Check symmetry for a few random pairs
    np.random.seed(42)
    for _ in range(5):
        i = np.random.randint(0, len(random_data))
        j = np.random.randint(0, len(random_data))
        if i == j:
            continue
            
        # Find distance from i to j
        dist_i_to_j = None
        for neighbor, dist in adjacency_list[i]:
            if neighbor == j:
                dist_i_to_j = dist
                break
                
        # Find distance from j to i
        dist_j_to_i = None
        for neighbor, dist in adjacency_list[j]:
            if neighbor == i:
                dist_j_to_i = dist
                break
                
        # Distances should be equal
        assert dist_i_to_j is not None
        assert dist_j_to_i is not None
        assert np.isclose(dist_i_to_j, dist_j_to_i)

# Test with edge cases
def test_k_equals_one():
    """Test when k=1 (only one neighbor per point)."""
    data = np.random.rand(10, 2)
    adjacency_list = build_neighbor_graph(data, 1)
    
    # Check that each point has exactly 1 neighbor
    for neighbors in adjacency_list:
        assert len(neighbors) == 1

def test_k_equals_n_minus_one():
    """Test when k=N-1 (all points except self)."""
    data = np.random.rand(5, 2)
    k = len(data) - 1
    adjacency_list = build_neighbor_graph(data, k)
    
    # Check that each point has exactly N-1 neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k
        
    # Check that no point is its own neighbor
    for i, neighbors in enumerate(adjacency_list):
        neighbor_indices = [n[0] for n in neighbors]
        assert i not in neighbor_indices

def test_small_dataset():
    """Test with a very small dataset."""
    data = np.random.rand(3, 2)
    k = 2
    adjacency_list = build_neighbor_graph(data, k)
    
    # Check that each point has exactly 2 neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

def test_high_dimensional_data():
    """Test with high-dimensional data."""
    data = np.random.rand(10, 10)  # 10 points in 10D
    k = 3
    adjacency_list = build_neighbor_graph(data, k)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k

def test_duplicate_points():
    """Test behavior with duplicate points in the dataset."""
    # Create data with some duplicate points
    data = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],  # Duplicate of point 0
        [2.0, 2.0],
        [1.0, 1.0],  # Duplicate of point 1
    ], dtype=np.float64)
    
    k = 2
    adjacency_list = build_neighbor_graph(data, k)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k
        
    # Point 0 should have point 2 as its closest neighbor (distance 0)
    neighbors_0 = adjacency_list[0]
    assert any(neighbor[0] == 2 and np.isclose(neighbor[1], 0.0) for neighbor in neighbors_0)
    
    # Point 1 should have point 4 as its closest neighbor (distance 0)
    neighbors_1 = adjacency_list[1]
    assert any(neighbor[0] == 4 and np.isclose(neighbor[1], 0.0) for neighbor in neighbors_1)

# Test with different data types
def test_integer_data():
    """Test with integer data."""
    # Convert integer data to float64
    data = np.random.randint(0, 100, size=(10, 3)).astype(np.float64)
    k = 3
    adjacency_list = build_neighbor_graph(data, k)
    
    # Check that each point has exactly k neighbors
    for neighbors in adjacency_list:
        assert len(neighbors) == k 