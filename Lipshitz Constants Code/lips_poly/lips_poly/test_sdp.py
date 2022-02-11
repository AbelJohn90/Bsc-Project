import pdb
import cvxpy as cp
import numpy as np
import polyopt as po
import torch

from polyopt import utils
from mosek.fusion import (
        Domain, Model, Matrix, ObjectiveSense, Expr)
from timeit import default_timer as timer


layer_configs = [(20, 20, 1), ]
#layer_configs = [(786, 500, 1), ]
repeats = 1


def run_sdp(layer_configs):
    network = utils.fc(layer_configs[0])
    weights, biases = utils.weights_from_pytorch(network)
    fc = po.FullyConnected(weights, biases)
    # cost = po._sdp_cost(weights[0], weights[1])
    # weights_ = [np.array([[1, 0], [0, 1]]), np.array([[1, 1]])]
    cost = po._sdp_cost(*weights)
    M = Model('sdp')
    X = M.variable('X', Domain.inPSDCone(cost.shape[0]))
    C = Matrix.dense(1 / 4 * cost)
    ones = [1.0] * cost.shape[0]
    M.constraint('c1', Expr.sub(ones, X.diag()), Domain.greaterThan(0.0))
    M.objective(ObjectiveSense.Maximize, Expr.dot(C, X))
    M.setSolverParam("intpntCoTolRelGap", 1.0e-12)
    start = timer()
    M.solve()
    end = timer()
    print('time elapsed: ', end - start)
    # lbp = po.lower_bound_product(fc.weights, p=1)
    lbp = po.lower_bound_product(weights, p=1)
    lbs = po.lower_bound_sampling(
         network, layer_configs[0][0], n=50000, p=1)
    # ub_sdp = M.dualObjValue() / 4
    ub_sdp = M.dualObjValue()
    status = M.getDualSolutionStatus()
    x_np = M.getVariable('X').level()

    print('SDP BOUND: ', ub_sdp)
    print('SDP STATUS: ', status)
    print('LOWER BOUND PRODUCT: ', lbp)
    print('LOWER BOUND SAMPLING: ', lbs)


def run_sdp_cvxpy(layer_config):
    network = utils.fc(layer_configs[0])
    weights, biases = utils.weights_from_pytorch(network)
    C = po._sdp_cost(*weights) / 4
    n = C.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i, i] <= 1. for i in range(n)
    ]
    prob = cp.Problem(cp.Maximize(cp.trace(C @ X)),
                      constraints)
    start = timer()
    prob.solve(solver=cp.SCS)
    end = timer()
    print('time elapsed: ', end - start)
    # lbp = po.lower_bound_product(fc.weights, p=1)
    print("The optimal value is", prob.value)


def main():
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = run_sdp(layer_configs)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    results = run_sdp_cvxpy(layer_configs)
    print(results)


if __name__ == '__main__':
    main()

