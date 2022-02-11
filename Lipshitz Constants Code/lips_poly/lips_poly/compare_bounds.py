import pdb
import cvxpy as cp
import numpy as np
import polyopt as po
import torch

from polyopt import utils
from mosek.fusion import (
                          Domain, Model, Matrix, ObjectiveSense, Expr)
from timeit import default_timer as timer

import matlab.engine
import ast, collections, pickle
import json

#layer_configs = [(10,10,10,1), (20,20,10,1), (40,40,10,1)]
layer_configs = [(40,40,10,1)]
#layer_configs = [(640, 640,1)]
#layer_configs = [(10, 10, 1),(20, 20, 1),(40, 40, 1),(80, 80, 1),(160, 160, 1),(320, 320, 1),(640, 640, 1)]
#layer_configs = [(786, 500, 1), ]
repeats = 10
degs = [4]
#degs = [4]
sparsities = [15]
#sparsities = [5,10,15,20] # max number of connections
#algs = ['lp', 'sdp', 'sdp_l2', 'lbs_l2', 'lbs', 'lbp']
algs = ['lp']

if 'sdp_l2' in algs:
    eng = matlab.engine.start_matlab()

def sdp_ub_l2_two_layers(weights):
    W0 = matlab.double([list(w) for w in weights[0]])
    W1 = matlab.double(list(weights[1][0]))
    return eng.anonbase_lipschitz(W0,W1,float(weights[0].shape[1]),float(weights[1].shape[1]),False)

def sdp_ub(weights, biases):
    n_hidden_layers = len(weights) - 1
    fc = po.FullyConnected(weights, biases)
    cost = po._sdp_cost(weights)
    M = Model('sdp')
    C = Matrix.dense(1 / (2**(1+n_hidden_layers)) * cost)
    X = M.variable('X', Domain.inPSDCone(cost.shape[0]))
    ones = [1.0] * cost.shape[0]
    M.constraint('c1', Expr.sub(ones, X.diag()), Domain.greaterThan(0.0))
    if n_hidden_layers == 2:
        d = weights[0].shape[1]
        n1 = weights[1].shape[1]
        n2 = weights[2].shape[1]
        for i in range(n1):
            for j in range(n2):
                M.constraint(Expr.sub(X.index([1+d+i,1+d+n1+j]),
                                      X.index([1+d+n1+n2 + n2*i + j,cost.shape[0]-1])), Domain.equalsTo(0.0))
    M.objective(ObjectiveSense.Maximize, Expr.dot(C, X))
    M.setSolverParam("intpntCoTolRelGap", 1.0e-12)
    start = timer()
    M.solve()
    end = timer()
    print('time elapsed: ', end - start)
    
    return M.dualObjValue()

def lp_ub(weights, biases, layer_config, deg=3, decomp='', sparsity=-1, reload=False, filename=""):
    fc = po.FullyConnected(weights, biases)
    f = fc.grad_poly
    g = fc.krivine_constr(p=1)
    start_indices = fc.start_indices
    variables = fc.variables
    
    m = po.KrivineOptimizer.maximize(f, g, deg=deg, start_indices=start_indices, layer_config=layer_config, solver='gurobi', decomp=decomp, sparsity=sparsity, weights=weights, n_jobs=-1, name='', reload=reload, use_filename=filename)
    return m.objVal

def main():
    filename = ""
    for layer_config in layer_configs:
        for sparsity in sparsities:
            for deg in degs:
                print("Run config {}, s = {}, deg = {}".format(layer_config, sparsity, deg))
                #ub_sdp_l2_list = []
                #ub_sdp_list = []
                #ub_lp_list = []
                #lb_sampling_l2_list = []
                #lb_sampling_list = []
                #lp_product_list = []
                bounds = dict()
                for alg in algs:
                    bounds[alg] = []
                reload = False # Compute 1st matrix, but not others
                for i in range(repeats):
                    np.random.seed(i)
                    torch.manual_seed(i)
                    network = utils.fc(layer_config, sparsity=sparsity)
                    weights, biases = utils.weights_from_pytorch(network)
                    
                    for alg in algs:
                        if alg == 'sdp_l2':
                            bound = sdp_ub_l2_two_layers(weights)
                            print('SDP BOUND L2: ', bound)
                            print('SDP BOUND L2 * sqrt(d): ', bound * np.sqrt(layer_config[0]))
                        elif alg == 'sdp':
                            bound = sdp_ub(weights, biases)
                            print('SDP BOUND: ', bound)
                        elif alg == 'lp':
                            bound = lp_ub(weights, biases, layer_config, deg=deg, decomp="multi layers", sparsity=sparsity, reload=reload, filename="")
                            reload = False
                            print('LP BOUND: ', bound)
                        elif alg == 'lbs_l2':
                            bound = po.lower_bound_sampling(network, layer_config[0], n=50000, p=2)
                            print('LOWER BOUND SAMPLING L2: ', bound)
                        elif alg == 'lbs':
                            bound = po.lower_bound_sampling(network, layer_config[0], n=50000, p=1)
                            print('LOWER BOUND SAMPLING: ', bound)
                        elif alg == 'lbp':
                            bound = po.lower_bound_product(weights, p=1)
                            print('LOWER BOUND PRODUCT: ', bound)
                        elif alg == 'ubp':
                            bound = po.upper_bound_product(weights, p=1)
                            print('UPPER BOUND PRODUCT: ', bound)
                        bounds[alg].append(float(bound))
                
                for alg in algs:
                    print("AVERAGE BOUND {}: {}".format(alg, np.mean(bounds[alg])))
                    s = ""
                    if sparsity > 0:
                        s = "_s={}".format(sparsity)
                    if alg == 'lp':
                        filename = "Results/bounds_lp_{}_deg={}{}.txt".format(layer_config, deg, s)
                    else:
                        filename = "Results/bounds_{}_{}{}.txt".format(alg, layer_config, s)
                    with open(filename, 'w') as file:
                        file.write(json.dumps(bounds[alg]))
                #print('SDP AVERAGE BOUND L2 * sqrt(d): ', np.mean(ub_sdp_l2_list) * np.sqrt(layer_config[0]))
                #print('SDP AVERAGE BOUND: ', np.mean(ub_sdp_list))
                #print('LP AVERAGE BOUND: ', np.mean(ub_lp_list))
                #print('LOWER AVERAGE BOUND PRODUCT: ', np.mean(lp_product_list))
                #print('LOWER AVERAGE BOUND SAMPLING: ', np.mean(lb_sampling_list))


if __name__ == '__main__':
    main()
