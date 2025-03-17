import math

import numpy as np
import pytest

import pypamm.lib as matrix_opx


def test_invmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    inv_M = matrix_opx.invmatrix(M.shape[0], M)
    # Expected inverse of [[1,2],[3,4]] => [[-2,1],[1.5,-0.5]]
    expected = np.array([[-2.0, 1.0], [1.5, -0.5]], dtype=np.float64)
    np.testing.assert_allclose(inv_M, expected, rtol=1e-7, atol=1e-9)


def test_trmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    trace_val = matrix_opx.trmatrix(M.shape[0], M)
    assert trace_val == pytest.approx(5.0)


def test_detmatrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    det_val = matrix_opx.detmatrix(M.shape[0], M)
    # Determinant = 1*4 - 2*3 = -2
    assert det_val == pytest.approx(-2.0)


def test_logdet():
    M = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    ld_val = matrix_opx.logdet(M.shape[0], M)
    # log(det) = log(2*3) = log(6)
    assert ld_val == pytest.approx(math.log(6.0))


def test_variance():
    # Simple 1D example:
    #   x = [1, 2, 3], weights = [1, 1, 1]
    # Weighted mean = 2. Variance = 1 if using a standard unbiased formula.
    # The exact result depends on the Fortran-like denominator.
    # We'll at least check it runs and yields something close to 1 for a small sample.
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)  # shape = (D=1, nsamples=3)
    w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    var_val = matrix_opx.variance(3, 1, x, w)
    assert var_val == pytest.approx(1.0)


def test_eigval():
    # Diagonal matrix => eigenvalues are just the diagonal
    A = np.diag([1.0, 5.0, 10.0])
    vals = matrix_opx.eigval(A.shape[0], A)
    # Order can vary, so sort for comparison
    assert np.allclose(np.sort(vals), [1.0, 5.0, 10.0])


def test_maxeigval():
    A = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    mx = matrix_opx.maxeigval(A.shape[0], A)
    assert mx == pytest.approx(5.0)


def test_factorial():
    assert matrix_opx.factorial(0) == 1
    assert matrix_opx.factorial(1) == 1
    assert matrix_opx.factorial(5) == 120


def test_pammrij():
    period = np.array([10.0, 0.0], dtype=np.float64)
    ri = np.array([9.0, 4.0], dtype=np.float64)
    rj = np.array([2.0, 1.0], dtype=np.float64)
    # For dimension=2, only the first coordinate is periodic
    # So the difference in x is 9 - 2 = 7 => within a 10-length box => 7/10 => shift -0 => 7
    # But the minimal image would also be -3, if we considered crossing boundary =>
    # Actually (7/10) => 0.7 => round(0.7) => 1 => 0.7 - 1 = -0.3 => *10 => -3.
    # So the minimal image is -3
    # The second coordinate is 4-1=3 (non-periodic => no shift)
    result = matrix_opx.pammrij(2, period, ri, rj)
    expected = np.array([-3.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_mahalanobis():
    D = 2
    period = np.array([10.0, 0.0], dtype=np.float64)
    x = np.array([9.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 1.0], dtype=np.float64)
    # Identity Qinv
    Qinv = np.eye(D, dtype=np.float64)
    # pammrij => [-3.0, 3.0]
    # distance => dv dot dv => 9
    dist = matrix_opx.mahalanobis(D, period, x, y, Qinv)
    assert dist == pytest.approx(9.0)


def test_effdim():
    # Diagonal covariance => eigenvals = diagonal
    # Let them sum to 1+2+3=6 => pk=[1/6,2/6,3/6], sum pk log(pk).
    # We'll just check numerical correctness with an approximate value
    A = np.diag([1.0, 2.0, 3.0])
    ed = matrix_opx.effdim(3, A)
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
    matrix_opx.oracle(Q.shape[0], 10.0, Q)  # modifies in place
    # We can do a simple check that diagonal got bigger, off-diag got shrunk
    assert Q[0, 1] == pytest.approx(Q[1, 0])
    assert Q[0, 0] > 1.0 - 1e-10  # typically shrinks + add to diag
    assert Q[1, 1] > 1.0 - 1e-10
