import numpy as np
import pytest

from pypamm.grid_selection import select_grid_points
from pypamm.lib.neighbours import compute_voronoi, get_voronoi_neighbour_list


def test_voronoi_basic(simple_data_voronoi):
    X, Y, wj, idxgrid = simple_data_voronoi
    iminij, ni, wi, ineigh = compute_voronoi(X, wj, Y, idxgrid, metric="euclidean")

    assert iminij.shape == (X.shape[0],)
    assert ni.shape == (Y.shape[0],)
    assert wi.shape == (Y.shape[0],)
    assert ineigh.shape == (Y.shape[0],)
    assert np.sum(ni) == X.shape[0]
    np.testing.assert_allclose(np.sum(wi), np.sum(wj))


def test_voronoi_single_grid_point(random_2d_data):
    wj = np.zeros(random_2d_data.shape[0], dtype=np.float64)

    Y = np.array([[0.5, 0.5]], dtype=np.float64)
    idxgrid = np.array([0], dtype=np.int32)

    iminij, ni, wi, ineigh = compute_voronoi(random_2d_data, wj, Y, idxgrid, metric="euclidean")

    assert np.all(iminij == 0)
    assert ni[0] == random_2d_data.shape[0]
    assert np.isclose(wi[0], np.sum(wj))


def test_voronoi_zero_weights(random_2d_data):
    wj = np.zeros(random_2d_data.shape[0], dtype=np.float64)
    Y = np.array([[0.0, 0.0], [1.0, 1.0]])
    idxgrid = np.array([0, 1], dtype=np.int32)

    iminij, ni, wi, ineigh = compute_voronoi(random_2d_data, wj, Y, idxgrid, metric="euclidean")

    assert np.sum(ni) == random_2d_data.shape[0]
    assert np.allclose(wi, 0.0)
    assert ineigh.shape == (Y.shape[0],)


def test_voronoi_invalid_metric(simple_data_voronoi):
    X, Y, wj, idxgrid = simple_data_voronoi

    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_voronoi(X, wj, Y, idxgrid, metric="banana")


def test_grid_selection_voronoi(random_2d_data):
    wj = np.zeros(random_2d_data.shape[0], dtype=np.float64)

    idxgrid, Y = select_grid_points(random_2d_data, 10, metric="euclidean")

    iminij, ni, wi, ineigh = compute_voronoi(random_2d_data, wj, Y, idxgrid, metric="euclidean")

    assert iminij.shape == (random_2d_data.shape[0],)
    assert ni.shape == (Y.shape[0],)
    assert wi.shape == (Y.shape[0],)
    assert ineigh.shape == (Y.shape[0],)
    assert np.sum(ni) == random_2d_data.shape[0]
    np.testing.assert_allclose(np.sum(wi), np.sum(wj))


# Test the neighbor list
def test_get_voronoi_neighbour_list_basic():
    nsamples = 6
    ngrid = 3

    # Each sample assigned to a Voronoi cell (0, 1, or 2)
    iminij = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    # Number of samples per Voronoi cell
    ni = np.array([2, 2, 2], dtype=np.int32)

    pnlist, nlist = get_voronoi_neighbour_list(nsamples, ngrid, ni, iminij)

    # Expected: [0, 2, 4, 6] → cell 0: nlist[0:2], cell 1: nlist[2:4], cell 2: nlist[4:6]
    assert pnlist.tolist() == [0, 2, 4, 6]

    # Check that neighbors are correctly assigned (order within segment can vary)
    for cell in range(ngrid):
        assigned = [i for i in range(nsamples) if iminij[i] == cell]
        actual = nlist[pnlist[cell] : pnlist[cell + 1]].tolist()
        assert sorted(assigned) == sorted(actual)
