import math

import numpy as np
import pytest

import pypamm.lib as matrix_opx


def test_invmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    inv_M = matrix_opx.invmatrix(M)
    # Expected inverse of [[1,2],[3,4]] => [[-2,1],[1.5,-0.5]]
    expected = np.array([[-2.0, 1.0], [1.5, -0.5]], dtype=np.float64)
    np.testing.assert_allclose(inv_M, expected, rtol=1e-7, atol=1e-9)


def test_trmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    trace_val = matrix_opx.trmatrix(M)
    assert trace_val == pytest.approx(5.0)


def test_detmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    det_val = matrix_opx.detmatrix(M)
    # Determinant = 1*4 - 2*3 = -2
    assert det_val == pytest.approx(-2.0)


def test_logdet():
    M = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    ld_val = matrix_opx.logdet(M)
    # log(det) = log(2*3) = log(6)
    assert ld_val == pytest.approx(math.log(6.0))


def test_eigval():
    # Diagonal matrix => eigenvalues are just the diagonal
    A = np.diag([1.0, 5.0, 10.0])
    vals = matrix_opx.eigval(A)
    # Order can vary, so sort for comparison
    assert np.allclose(np.sort(vals), [1.0, 5.0, 10.0])


def test_maxeigval():
    A = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    mx = matrix_opx.maxeigval(A)
    assert mx == pytest.approx(5.0)


def test_factorial():
    assert matrix_opx.factorial(0) == 1
    assert matrix_opx.factorial(1) == 1
    assert matrix_opx.factorial(5) == 120


def test_effdim():
    # Diagonal covariance => eigenvals = diagonal
    # Let them sum to 1+2+3=6 => pk=[1/6,2/6,3/6], sum pk log(pk).
    # We'll just check numerical correctness with an approximate value
    A = np.diag([1.0, 2.0, 3.0])
    ed = matrix_opx.effdim(A)
    # Let's compute reference: pk=[1/6, 2/6, 3/6].
    # sum_plogp= (1/6)*log(1/6)+(2/6)*log(2/6)+(3/6)*log(3/6).
    # effdim= exp(-sum_plogp). We'll just compare to that numeric result.
    pk = np.array([1.0 / 6, 2.0 / 6, 3.0 / 6], dtype=np.float64)
    sum_plogp = np.sum(pk * np.log(pk))
    expected = np.exp(-sum_plogp)
    assert ed == pytest.approx(expected)


def test_oracle():
    # A small 2x2 example
    Q = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)
    # We pass Q in by reference, so it gets modified in place
    # Just ensure it runs and modifies Q in some consistent way
    matrix_opx.oracle(Q, 10.0)  # modifies in place
    # We can do a simple check that diagonal got bigger, off-diag got shrunk
    assert Q[0, 1] == pytest.approx(Q[1, 0])
    assert Q[0, 0] > 1.0 - 1e-10  # typically shrinks + add to diag
    assert Q[1, 1] > 1.0 - 1e-10


def test_covariance_against_numpy():
    """Test weighted covariance matches numpy's cov with weights."""
    np.random.seed(0)
    X = np.random.rand(100, 3)
    w = np.random.rand(100)
    w /= w.sum()  # normalize to sum = 1
    wnorm = 1.0

    # Run your Cython version
    Q = matrix_opx.wcovariance(X, w, wnorm)

    # Compute weighted mean and cov using NumPy
    xm = np.average(X, axis=0, weights=w)
    Xc = X - xm
    Xw = Xc * (w[:, None])  # broadcast weights
    Q_np = Xc.T @ Xw / (1.0 - np.sum(w**2))

    # Check shapes and numerical closeness
    assert Q.shape == Q_np.shape
    np.testing.assert_allclose(Q, Q_np, rtol=1e-10, atol=1e-12)
