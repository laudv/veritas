import os, io, pickle, time, timeit, subprocess, json, sys
import multiprocessing as mp
import scipy.io
import numpy as np
import xgboost as xgb

import util
from treeck import *
from treeck.xgb import addtree_from_xgb_model

RESULT_DIR = "tests/experiments/scale-mnist"

# - Loading the covertype data set --------------------------------------------

X, y = util.load_openml("mnist", data_id=554)
y = (y==2)
num_examples, num_features = X.shape
print("balance:", sum(y) / num_examples)
Itrain, Itest = util.train_test_indices(num_examples)



# - Training XGBoost model ----------------------------------------------------

def train_model(lr, num_trees, max_depth=5):
    model_name = f"model-{num_trees}.xgb"
    dtest = xgb.DMatrix(X[Itest], y[Itest], missing=None)
    if not os.path.isfile(os.path.join(RESULT_DIR, model_name)):
        print(f"training model learning_rate={lr}, num_trees={num_trees}")
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": max_depth,
            "learning_rate": lr,
            "eval_metric": "error",
            "seed": 14,
        }

        dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
        model = xgb.train(params, dtrain, num_boost_round=num_trees,
                          #early_stopping_rounds=5,
                          evals=[(dtrain, "train"), (dtest, "test")])
        with open(os.path.join(RESULT_DIR, model_name), "wb") as f:
            pickle.dump(model, f)
        #with open(os.path.join(RESULT_DIR, "model.json"), "w") as f:
        #    model.dump_model(f, dump_format="json")
    else:
        print(f"loading model from file: {model_name}")
        with open(os.path.join(RESULT_DIR, model_name), "rb") as f:
            model = pickle.load(f)

    yhat = model.predict(dtest) > 0.5
    acc = sum(y[Itest] != yhat) / X.shape[0]

    return model, acc


# - Optimizer routines --------------------------------------------------------

def get_opt():
    opt = Optimizer(maximize=at, max_memory=MAX_MEMORY)
    #opt.enable_smt()
    feat_ids = opt.get_used_feat_ids()[1]

    #smt = io.StringIO()
    #print(f"(assert (> {opt.xvar(1, 0)} 3200.0))", file=smt) # elevation
    #print(f"(assert (< {opt.xvar(1, 5)} 1800.0))", file=smt) # hoz dist to road
    ##print(f"(assert (> {opt.xvar(1, 9)} 1800.0))", file=smt) # hoz dist fire road
    #for i in set(range(10, 14)).intersection(feat_ids):
    #    op = ">" if i == 13 else "<"
    #    print(f"(assert ({op} {opt.xvar(1, i)} 0.5))", file=smt) # Wilderness_Area
    #for i in set(range(14, 54)).intersection(feat_ids):
    #    op = ">" if i == 36 else "<"
    #    print(f"(assert ({op} {opt.xvar(1, i)} 0.5))", file=smt) # Soil_Type

    #opt.set_smt_program(smt.getvalue())

    print("before num_vertices", opt.num_vertices(1))
    #opt.prune()
    print("after num_vertices", opt.num_vertices(1))
    #opt.disable_smt()

    if USE_DYN_PROG:
        opt.use_dyn_prog_heuristic()
    return opt

def astar(mergeK = 0):
    nsteps = 100
    opt = get_opt()

    timings = [0.0]
    bounds = [opt.current_bounds()[1]]
    memory = [opt.memory()]
    steps = [0]

    start = timeit.default_timer()
    if mergeK > 0:
        opt.merge(mergeK)
        timings.append(timeit.default_timer() - start)
        bounds.append(opt.current_bounds()[1])
        memory.append(opt.memory())
        steps.append(0)

    done = False
    while not done and opt.num_solutions() == 0:
        try:
            done = not opt.step(nsteps)
            nsteps = min(204800, nsteps * 2)
        except:
            print("OUT OF MEMORY")
            done = True

        timings.append(timeit.default_timer() - start)
        bounds.append(opt.current_bounds()[1])
        memory.append(opt.memory())
        steps.append(opt.nsteps()[1])

    return opt, timings, bounds, memory, steps

def arastar(eps, eps_incr, mergeK = 0):
    nsteps = 100
    opt = get_opt()

    timings = []
    memory = []
    steps = []

    start = timeit.default_timer()
    if mergeK > 0:
        opt.merge(mergeK)
        timings.append(timeit.default_timer() - start)
        memory.append(opt.memory())
        steps.append(opt.nsteps()[1])
    opt.set_ara_eps(eps, eps_incr)

    done = False
    while not done and opt.get_ara_eps() < 1.0:
        try:
            done = not opt.step(nsteps)
            nsteps = min(204800, nsteps * 2)
        except:
            print("OUT OF MEMORY")
            done = True

        while len(timings) < opt.num_solutions():
            timings.append(timeit.default_timer() - start)
            memory.append(opt.memory())
            steps.append(opt.nsteps()[1])

    return opt, timings, memory, steps

def merge(conn):
    opt = get_opt()

    t = 0.0
    b = opt.current_bounds()[1]
    m = opt.memory()
    v = opt.num_vertices(1)
    conn.send((t, b, m, v))

    start = timeit.default_timer()
    try:
        while True:
            try:
                opt.merge(2)
            except:
                print("MERGE worker: OUT OF MEMORY")
                break

            if b == opt.current_bounds()[1]:
                break

            t = timeit.default_timer() - start
            b = opt.current_bounds()[1]
            m = opt.memory()
            v = opt.num_vertices(1)
            conn.send((t, b, m, v))
    finally:
        print("MERGE worker: closing")
        conn.close()

def merge_in_process(max_runtime):
    cparent, cchild = mp.Pipe()
    p = mp.Process(target=merge, name="Merger", args=(cchild,))
    p.start()
    start = timeit.default_timer()
    timings, bounds, memory, vertices = [], [], [], []
    print("MERGE host: runtime", max_runtime)
    while timeit.default_timer() - start < max_runtime:
        has_data = cparent.poll(1)
        if has_data:
            t, b, m, v = cparent.recv()
            print("MERGE host: data", t, b, m, v)
            timings.append(t)
            bounds.append(b)
            memory.append(m)
            vertices.append(v)
        elif p.exitcode is not None:
            break
    print("MERGE host: terminating")
    p.terminate()
    cparent.close()
    return timings, bounds, memory, vertices


# - Robustness for increasing model complexity --------------------------------

MAX_MEMORY = 1024*1024*1024 * 2 # GB
USE_DYN_PROG = False

if __name__ == "__main__":
    o = []
    output_file = sys.argv[1]
    MAX_MEMORY = 1024*1024*1024*int(sys.argv[2])

    print("writing output to", os.path.join(RESULT_DIR, output_file))
    if input("OK? ") != "y":
        sys.exit()

    for lr, num_trees in [
            #(1.0, 1),
            #(0.9, 5),
            #(0.8, 10),
            #(0.75, 20),
            #(0.6, 30),
            #(0.65, 40),
            #(0.6, 50),
            (0.50, 60),
            ##(0.40, 70),
            #(0.35, 80),
            ##(0.30, 90),
            #(0.30, 100),
            ##(0.25, 110),
            #(0.25, 120),
            ##(0.20, 130),
            #(0.20, 140),
            ##(0.20, 150),
            #(0.15, 160),
            ##(0.15, 170),
            #(0.15, 180),
            ##(0.10, 190),
            #(0.10, 200),
            ]:

        #for i, (m0, m1) in enumerate(zip(X[y==0].mean(axis=0), X[y==1].mean(axis=0))):
        #    print(i, m0, m1, m1-m0)
        #break

        print(f"\n= num_trees {num_trees} ===========")
        model, acc = train_model(lr, num_trees)
        print(f"accuracy: {acc}")
        at = addtree_from_xgb_model(model)
        at.base_score = 0

        print("double check:", util.double_check_at_output(model, at, X[0:100, :]))

        start = timeit.default_timer()
        opt = get_opt()
        oo = {
            "num_vertices": opt.num_vertices(1),
            "num_trees": num_trees,
            "lr": lr,
        }
        del opt

        print(f"\n -- A* {time.ctime()} --")
        opt, t0, bounds0, m0, steps0 = astar(mergeK=0)
        oo["astar"] = {
            "solutions": [s[1] for s in opt.solutions()],
            "timings": t0,
            "bounds": bounds0,
            "memory": m0,
            "steps": steps0,
        }
        del opt

        print(f"\n -- ARA* {time.ctime()} --")
        opt, t1, m1, steps1 = arastar(0.2, 0.01, mergeK=0)
        oo["arastar"] = {
            "solutions": [s[1] for s in opt.solutions()],
            "timings": t1,
            "epses": opt.epses(),
            "memory": m1,
            "steps": steps1,
        }
        del opt

        print(f"\n -- MERGE {time.ctime()} --")
        t2, bounds2, m2, v2 = merge_in_process(max(10, (timeit.default_timer() - start)/1.8))
        oo["merge"] = {
            "timings": t2,
            "bounds": bounds2,
            "memory": m2,
            "vertices": v2,
        }

        o.append(oo)

    with open(os.path.join(RESULT_DIR, output_file), "w") as f:
        json.dump(o, f)
