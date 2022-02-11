import pytest
import cvxpy as cp
import numpy as np


@pytest.fixture
def sdp():
    n = 10
    p = 12
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    return [n, p, C, A, b]


def test_scs(sdp):
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    n, p, C, A, b = sdp
    X = cp.Variable((n, n), PSD=True)
    # The operator >> denotes matrix inequality.
    constraints = []
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.SCS, verbose=True)
    assert prob.value

