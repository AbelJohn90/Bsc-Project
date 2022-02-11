import pytest
import numpy as np
import sympy as sp

import polyopt as po


@pytest.fixture
def fc():
    w_0 = np.array([[1, 2, 3], [-1, 1, 2]])
    w_1 = np.array([[1, -2]])
    b_0 = np.array([0, 0])
    b_1 = np.array([0])
    return po.FullyConnected(
            weights=[w_0, w_1], biases=[b_0, b_1])


@pytest.fixture
def sparse_fc():
    w_0 = np.array([[1, 2, 3, 0.], [0., 0., 1, 2], [0., 0., -1, 1]])
    w_1 = np.array([[1, -2, 1.]])
    b_0 = np.array([0, 0., 0.])
    b_1 = np.array([0])
    return po.FullyConnected(
            weights=[w_0, w_1], biases=[b_0, b_1])


@pytest.fixture
def sparse_fc2():
    w_0 = np.array([
        [-2.3, 1.1, 3.4, 0.], [0., 0., 1.3, 4.1], [0., 0., -1.2, 1.3]])
    w_1 = np.array([[1.5, -2.3, 1.1]])
    b_0 = np.array([0, 0., 0.])
    b_1 = np.array([0])
    return po.FullyConnected(
            weights=[w_0, w_1], biases=[b_0, b_1])


def test_attributes(fc):
    assert fc.n_layers == 2
    assert fc.n_hidden_layers == 1
    for x in fc.vars_layer:
        assert x is not None
    assert len(fc.vars_layer[0]) == 3
    assert len(fc.vars_layer[1]) == 2


def test_gradpoly(fc):
    x = sp.symbols('x_0:5')
    expr = (
        x[0]*x[3] + 2*x[1]*x[3] + 3*x[2]*x[3]
        + 2*x[0]*x[4] - 2*x[1]*x[4] - 4*x[2]*x[4]
    )
    assert sp.Poly(expr) == sp.Poly(fc.grad_poly)


def test_optimize(fc):
    f = fc.grad_poly
    g = fc.krivine_constr(p=1)
    variables = fc.variables
    m = po.KrivineOptimizer.maximize(
            f, g, variables, deg=2, solver='gurobi')
    assert m.objVal == pytest.approx(8.)

def test_optimize2(fc):
    f = fc.grad_poly
    g = fc.krivine_constr(p=1)
    variables = fc.variables
    m = po.KrivineOptimizer.maximize2(
            f, g, variables, deg=2, solver='gurobi')
    assert m.objVal == pytest.approx(8.)


# def test_optimize_sparse(fc):
#     f = fc.grad_poly
#     g = fc._sparse_krivine_constr(p=1)
#     variables = fc.variables
#     m = po.KrivineOptimizer.maximize(
#             f, g, variables, deg=2, solver='gurobi')
#     assert m


def test_naive_linf_bound(fc):
    bound = po.naive_linf_bound(fc.weights, activ_lips=1.)
    assert bound == 18.


def test_preactivation_bound(fc):
    lb = np.array([0., 0., 0.])
    ub = np.array([1., 1., 1.])
    alb, aub = po._preactivation_bound(
            A=fc.weights[0], b=fc.biases[0], lb=lb, ub=ub)
    assert alb[0] == 0.
    assert alb[1] == -1.
    assert aub[0] == 6.
    assert aub[1] == 3.


def test_bounds(fc):
    lb = np.array([0., 0., 0.])
    ub = np.array([1., 1., 1.])
    bounds = po.bounds(
            weights=fc.weights, biases=fc.biases, lb=lb, ub=ub)
    assert bounds


def test_onelayer_sparsity_pattern(sparse_fc):
    result = po.onelayer_sparsity_pattern(sparse_fc.weights[0])
    assert result[0][0] == 0
    assert result[0][1] == 1
    assert result[0][2] == 2
    assert result[0][3] == 0

    assert result[1][0] == 2
    assert result[1][1] == 3
    assert result[1][2] == 1

    assert result[2][0] == 2
    assert result[2][1] == 3
    assert result[2][2] == 2


def test_sparse_krivine(sparse_fc, p=1):
    f = sparse_fc.grad_poly
    sparse_g = sparse_fc.sparse_krivine_constr()
    g = sparse_fc.krivine_constr()
    variables = sparse_fc.variables
    sparse_m = po.KrivineOptimizer.maximize(
            f, sparse_g, variables, deg=2, solver='gurobi', sparse=True)
    m = po.KrivineOptimizer.maximize(
            f, g, variables, deg=2, solver='gurobi', sparse=False)
    assert m.objVal == pytest.approx(sparse_m.objVal)


def test_sparse_krivine2(sparse_fc2, p=1):
    f = sparse_fc2.grad_poly
    sparse_g = sparse_fc2.sparse_krivine_constr()
    g = sparse_fc2.krivine_constr()
    variables = sparse_fc2.variables
    sparse_m = po.KrivineOptimizer.maximize(
            f, sparse_g, variables, deg=2, solver='gurobi', sparse=True)
    m = po.KrivineOptimizer.maximize(
            f, g, variables, deg=2, solver='gurobi', sparse=False)
    assert m.objVal == pytest.approx(sparse_m.objVal)

