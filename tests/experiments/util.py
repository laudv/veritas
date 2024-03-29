import os
import numpy as np
import pandas as pd
import scipy
import scipy.io
import xgboost as xgb
import math
from sklearn.datasets import fetch_openml

#from veritas import Optimizer, RealDomain

data_dir = "tests/data"

def load_openml(name, data_id, task="classification", force=False):
    """
    calhouse: data_id=537
    mnist: data_id=554
    covtype: data_id=1596
    """
    if not os.path.exists(f"{data_dir}/{name}.h5") or force:
        print(f"loading {name} with fetch_openml")
        X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
        #if task == "regression":
        #    y = np.array(list(map(float, d["target"])))
        #elif task == "classification":
        #    y = np.array(list(map(int, d["target"])))
        #else:
        #    raise RuntimeError("invalid task")
        #scipy.io.savemat(f"tests/data/{name}.mat", {"X": X, "y": y},
        #        do_compression=True, format="5")
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X.to_hdf(f"{data_dir}/{name}.h5", key="X", complevel=9)
        y.to_hdf(f"{data_dir}/{name}.h5", key="y", complevel=9)

    print(f"loading {name} h5 file")
    X = pd.read_hdf(f"{data_dir}/{name}.h5", key="X")
    y = pd.read_hdf(f"{data_dir}/{name}.h5", key="y")

    return X, y

def train_test_indices(num_examples, seed=82394188):
    np.random.seed(seed)
    indices = np.random.permutation(num_examples)

    m = int(num_examples*0.9)
    Itrain = indices[0:m]
    Itest = indices[m:]

    return Itrain, Itest

def optimize_learning_rate(X, y, params, num_trees, metric, seed=12419):
    num_examples, num_features = X.shape
    Itrain, Itest = train_test_indices(num_examples, seed=seed)
    if isinstance(X, pd.DataFrame):
        ytest = y.iloc[Itest]
        dtrain = xgb.DMatrix(X.iloc[Itrain], y.iloc[Itrain], missing=None)
        dtest = xgb.DMatrix(X.iloc[Itest], ytest, missing=None)
    else:
        ytest = y[Itest]
        dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
        dtest = xgb.DMatrix(X[Itest], ytest, missing=None)

    best_metric = -np.inf
    best_model = None
    best_lr = 0.0

    if "num_class" not in params:
        metric2 = lambda model: metric(ytest, model.predict(dtest, output_margin=True))
    else:
        metric2 = lambda model: metric(ytest, model.predict(dtest))

    lr_values_tested = set()
    for lr in np.linspace(0, 1, 5)[1:]:
        lr_values_tested.add(lr)
        print("(1) LEARNING_RATE =", lr)
        params["learning_rate"] = lr
        model = xgb.train(params, dtrain, num_boost_round=num_trees,
                          evals=[(dtrain, "train"), (dtest, "test")])
        m = metric2(model)
        print(f"(1) metric = {m}")
        if m > best_metric:
            print(f"(1) NEW BEST LEARNING_RATE {best_lr} -> {lr}")
            best_metric = m
            best_model = model
            best_lr = lr

    for lr in np.linspace(best_lr - 0.25, best_lr + 0.25, 7)[1:-1]:
        if lr <= 0.0 or lr > 1.0: continue
        if lr in lr_values_tested: continue
        print("(2) LEARNING_RATE =", lr)
        params["learning_rate"] = lr
        model = xgb.train(params, dtrain, num_boost_round=num_trees,
                          evals=[(dtrain, "train"), (dtest, "test")])
        m = metric2(model)
        print(f"(2) metric = {m}")
        if m > best_metric:
            print(f"(2) NEW BEST LEARNING_RATE {best_lr} -> {lr}")
            best_metric = m
            best_model = model
            best_lr = lr

    print("(-) BEST LEARNING_RATE =", best_lr, ", metric =", best_metric)

    return best_model, float(best_lr), float(best_metric)

def double_check_at_output(model, at, X):
    max_diff = 0.0
    tmp = model.predict(xgb.DMatrix(X, missing=None), output_margin=True)
    for i in range(X.shape[0]):
        p = at.predict_single(X[i, :])
        max_diff = max(max_diff, p-tmp[i])
    return max_diff

def get_ara_bound(epses, sols, task="maximize"):
    """
    epses strictly fall over time, but a solutions don't
    a solution s is eps optimal with eps the eps of the last solution that
    is smaller than s

    epses = [0.5 0.6 0.7 0.8]
    sols =  [5.0 4.4 3.9 6.3]
    """
    assert len(epses) == len(sols)
    solsbest = sols.copy()
    epsesbest = epses.copy()
    ibest = 0
    for i in range(1, len(sols) + 1):
        better = i == len(sols) \
              or (task == "maximize" and sols[ibest] < sols[i]) \
              or (task == "minimize" and sols[ibest] > sols[i])
        if better:
            old_ibest = ibest
            while ibest < i:
                solsbest[ibest] = sols[old_ibest] # ibest was better than everything until i
                epsesbest[ibest] = epses[i-1] # sols[old_ibest] is epses[i-1] optimal
                ibest += 1
            assert ibest == i
    bound = [s/e for s, e in zip(solsbest, epsesbest)]
    return solsbest, epsesbest, bound

def filter_solutions(*args):
    #if len(args) == 1 and isinstance(args[0], ParallelOptimizer):
    #    sols = []
    #    paropt = args[0]
    #    for i in range(paropt.num_threads()):
    #        sols += paropt.worker_opt(i).solutions
    #        #sols += [s for s in paropt.worker_opt(i).solutions if s.is_valid]
    #    sols.sort(key=lambda s: s.output_difference(), reverse=True)
    #    sols.sort(key=lambda s: s.eps) # stable sort

    #    fsols = [] # filtered solutions
    #    prev_eps = -1
    #    for s in sols:
    #        if s.eps != prev_eps:
    #            fsols.append(s)
    #        prev_eps = s.eps
    #    return fsols

    if len(args) == 1 and isinstance(args[0], Optimizer):
        opt = args[0]
        sols = opt.solutions()
        sols.sort(key=lambda s: s.output_difference(), reverse=True)
        sols.sort(key=lambda s: s.eps) # stable sort

        fsols = [] # filtered solutions
        prev_eps = -1
        for s in sols:
            if s.eps != prev_eps:
                fsols.append(s)
            prev_eps = s.eps
        return fsols
 
    if len(args) == 4: # (output0, output1, time, epses)
        sols = list(zip(*args))
        sols.sort(key=lambda x: x[1]-x[0], reverse=True)
        sols.sort(key=lambda x: x[3]) # stable sort
        prev_eps = -1
        s0, s1, ts, es = [], [], [], []
        for i in range(len(sols)):
            if sols[i][3] != prev_eps:
                a, b, t, e = sols[i]
                s0.append(a)
                s1.append(b)
                ts.append(t)
                es.append(e)
            prev_eps = sols[i][3]
        return s0, s1, ts, es

def flatten_ara_upper(ss, es):
    bs = []
    last_b = math.inf
    for s, e in zip(ss, es):
        b = s/e
        if b < last_b:
            bs.append(b)
            last_b = b
        else:
            bs.append(last_b)
    return bs

def get_best_astar(A, task="maximize"):
    if task == "maximize":
        if len(A["solutions"]) > 0:
            return A["solutions"][0][1]
        return A["bounds"][-1][1]
    elif task == "minimize":
        if len(A["solutions"]) > 0:
            return A["solutions"][0][0]
        return A["bounds"][-1][0]
    else:
        if len(A["solutions"]) > 0:
            return A["solutions"][0]
        return A["bounds"][-1]

def generate_random_constraints(X, num_constraints, seed):
    K = X.shape[1]
    m = X.min(axis=0)
    M = X.max(axis=0)

    rng = np.random.RandomState(seed)

    constraints = [RealDomain(m, M) for m, M in zip(m, M)]

    maybe_binary = [m == 0.0 and M == 1.0 for m, M in zip(m, M)]
    binary = [False for _ in range(K)]
    for k in range(K):
        if not maybe_binary[k]: continue
        if len(np.unique(X[0:100, k])) > 2: continue
        if len(np.unique(X[:, k])) == 2: binary[k] = True

    for k in rng.randint(0, K, num_constraints):
        if binary[k]:
            if rng.rand() < 0.5:
                constraints[k] = RealDomain(0.0, 0.5)
            else:
                constraints[k] = RealDomain(0.5, 1.0)
        else:
            c = constraints[k]
            split = c.lo + rng.rand() * (c.hi - c.lo)
            try:
                l, r = constraints[k].split(split)
            except:
                continue
            constraints[k] = l if rng.rand() < 0.5 else r
            #print(f"{c} -> {constraints[k]}")

    return constraints

def randomly_prune_opt(X, opt, seed):
    K = X.shape[1]
    m = X.min(axis=0)
    M = X.max(axis=0)

    rng = np.random.RandomState(seed)
    box = [RealDomain(m, M) for m, M in zip(m, M)]
    num_tries = 0

    fraction = max(0.05, rng.rand())
    num_vertices_goal = fraction * (opt.g0.num_vertices() + opt.g1.num_vertices())

    #print("prune start:", fraction, num_vertices_goal, (opt.g0.num_vertices() + opt.g1.num_vertices()))
    
    while opt.g0.num_vertices() + opt.g1.num_vertices() > num_vertices_goal and num_tries < 100:
        num_tries += 1
        for k in rng.randint(0, K, 10):
            c = box[k]
            split = c.lo + rng.rand() * (c.hi - c.lo)
            try:
                l, r = box[k].split(split)
            except:
                continue
            box[k] = l if rng.rand() < 0.5 else r
        opt.prune_box(box, 0)
        opt.prune_box(box, 1)

        #print("prune progress:", fraction, num_vertices_goal, (opt.g0.num_vertices() + opt.g1.num_vertices()))

    return box,num_vertices_goal
