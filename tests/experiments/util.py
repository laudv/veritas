import os
import numpy as np
import scipy
import scipy.io
import xgboost as xgb
import math
from sklearn.datasets import fetch_openml

from treeck import Optimizer, ParallelOptimizer

def load_openml(name, data_id, task="classification", force=False):
    """
    calhouse: data_id=537
    mnist: data_id=554
    covtype: data_id=1596
    """
    if not os.path.exists(f"tests/data/{name}.mat") or force:
        print(f"loading {name} with fetch_openml")
        d = fetch_openml(data_id=data_id)
        X = d["data"]
        if task == "regression":
            y = np.array(list(map(float, d["target"])))
        elif task == "classification":
            y = np.array(list(map(int, d["target"])))
        else:
            raise RuntimeError("invalid task")
        scipy.io.savemat(f"tests/data/{name}.mat", {"X": X, "y": y},
                do_compression=True, format="5")
    else:
        print(f"loading {name} MAT file")
        mat = scipy.io.loadmat(f"tests/data/{name}.mat") # much faster
        X = mat["X"]
        y = mat["y"]

    y = y.reshape((X.shape[0],))

    return X, y

def train_test_indices(num_examples, seed=82394188):
    np.random.seed(seed)
    indices = np.random.permutation(num_examples)

    m = int(num_examples*0.9)
    Itrain = indices[0:m]
    Itest = indices[m:]

    return Itrain, Itest

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
    if len(args) == 1 and isinstance(args[0], ParallelOptimizer):
        sols = []
        paropt = args[0]
        for i in range(paropt.num_threads()):
            sols += paropt.worker_opt(i).solutions
            #sols += [s for s in paropt.worker_opt(i).solutions if s.is_valid]
        sols.sort(key=lambda s: s.output_difference(), reverse=True)
        sols.sort(key=lambda s: s.eps) # stable sort

        fsols = [] # filtered solutions
        prev_eps = -1
        for s in sols:
            if s.eps != prev_eps:
                fsols.append(s)
            prev_eps = s.eps
        return fsols

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
