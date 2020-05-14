import os, pickle, io, timeit, subprocess
import sklearn.datasets as skds
import scipy
import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from treeck import *
from treeck.xgb import addtrees_from_multiclass_xgb_model

RESULT_DIR = "tests/experiments/mnist-single-instance"


# - Loading the MNIST data set ------------------------------------------------

if not os.path.exists("tests/data/mnist.mat"):
    print("loading MNIST with fetch_openml")
    mnist = fetch_openml("mnist_784")
    X = mnist["data"]
    y = np.array(list(map(lambda v: int(v), mnist["target"])))
    scipy.io.savemat("tests/data/mnist.mat", {"X": X, "y": y},
            do_compression=True, format="5")
else:
    print("loading MNIST MAT file")
    mat = scipy.io.loadmat("tests/data/mnist.mat") # much faster
    X = mat["X"]
    y = mat["y"].reshape((70000,))

num_examples = X.shape[0]
num_features = X.shape[1]

np.random.seed(112)
indices = np.random.permutation(num_examples)

m = int(num_examples*0.9)
Itrain = indices[0:m]
Itest = indices[m:]

if not os.path.exists("tests/data/mnist.libsvm"):
    skds.dump_svmlight_file(X[Itest[0:10]]+1, y[Itest[0:10]], "tests/data/mnist.libsvm")


# - Training an XGBoost model -------------------------------------------------

if not os.path.isfile(os.path.join(RESULT_DIR, "model.xgb")):
    print("training model")
    params = {
        "objective": "multi:softmax",
        "num_class": 10,
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.4,
        "eval_metric": "merror",
        "seed": 11,
    }

    dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
    dtest = xgb.DMatrix(X[Itest], y[Itest], missing=None)

    model = xgb.train(params, dtrain, num_boost_round=200,
                      early_stopping_rounds=5,
                      evals=[(dtrain, "train"), (dtest, "test")])
    with open(os.path.join(RESULT_DIR, "model.xgb"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(RESULT_DIR, "model.json"), "w") as f:
        model.dump_model(f, dump_format="json")
else:
    print("loading model from file")
    with open(os.path.join(RESULT_DIR, "model.xgb"), "rb") as f:
        model = pickle.load(f)

ats = addtrees_from_multiclass_xgb_model(model, 10)


# - Generate adversarial examples ---------------------------------------------

n_adv = 10
instances = X[Itest[0:n_adv]]
labels = y[Itest[0:n_adv]]

def set_smt(opt, instance, eps): # for minimization specifically
    smt = io.StringIO()
    for feat_id in opt.get_used_feat_ids()[0]:
        x = opt.xvar(0, feat_id)
        v = instance[feat_id]
        print(f"(assert (<= {x} {v+eps/2}))", file=smt)
        print(f"(assert (>= {x} {v-eps/2}))", file=smt)
    for feat_id in opt.get_used_feat_ids()[1]:
        x = opt.xvar(1, feat_id)
        v = instance[feat_id]
        print(f"(assert (<= {x} {v+eps/2}))", file=smt)
        print(f"(assert (>= {x} {v-eps/2}))", file=smt)
    opt.set_smt_program(smt.getvalue())

for i, (instance, label) in enumerate(zip(instances, labels)):
    at = ats[label]
    print(label)
    
    #opt = Optimizer(at, ats[0], set(), False)
    opt = Optimizer(at, minimize=True)
    eps = 20
    set_smt(opt, instance, eps=eps)
    
    bound_before = opt.current_bounds()
    num_vertices_before = opt.num_vertices(0)
    num_vertices_before1 = opt.num_vertices(1)
    start = timeit.default_timer()
    opt.prune()
    print("time prune:", timeit.default_timer()-start)
    num_vertices_after = opt.num_vertices(0)
    num_vertices_after1 = opt.num_vertices(1)
    opt.reset_smt_program()
    bound_after = opt.current_bounds()
    print("prune: num_vertices0 ", num_vertices_before, "->", num_vertices_after)
    print("       num_vertices1 ", num_vertices_before1, "->", num_vertices_after1)
    print("       bound         ", bound_before, "->", bound_after)

    num_vertices_before = opt.num_vertices(0)
    num_indep_sets_before = opt.num_independent_sets(0)
    bound_before = opt.current_bounds()
    opt.merge(2, instance=0)
    opt.merge(2, instance=1)
    opt.merge(2, instance=0)
    opt.merge(2, instance=1) # like in paper
    num_vertices_after = opt.num_vertices(0)
    num_indep_sets_after = opt.num_independent_sets(0)
    bound_after = opt.current_bounds()
    print("merge: num_vertices  ", num_vertices_before, "->", num_vertices_after)
    print("       num_indep_sets", num_indep_sets_before, "->", num_indep_sets_after)
    print("       bound         ", bound_before, "->", bound_after)

    bounds = [opt.current_bounds()]
    stepsize = 100

    start = timeit.default_timer()
    while opt.num_solutions() == 0 and opt.num_candidate_cliques() < 1000000:
        #if not opt.step(stepsize, 0.0, 0.0):
        if not opt.step(stepsize, 0.0):
            print("NO SOLUTION")
            break
        b = opt.current_bounds()
        #print(b, opt.nsteps(), opt.num_candidate_cliques())
        bounds.append(b)
    print("time step:", timeit.default_timer()-start)

    print("step: num_steps    ", opt.nsteps())
    print("      num_cliques  ", opt.num_candidate_cliques())
    print("      num_solutions", opt.num_solutions())
    print("      bound        ", bound_after, "->", bounds[-1] if len(bounds) > 0 else bound_after)

    # try treeVerify, assume installed in parent directory
    treeVerifyBin = "../treeVerification/treeVerify"
    with open(os.path.join(RESULT_DIR, "treeVerify.json"), "w") as f:
        print('{', file=f)
        print('    "inputs":       "tests/data/mnist.libsvm", ', file=f)
        print(f'    "model":        "{os.path.join(RESULT_DIR, "model.json")}",', file=f)
        print(f'    "start_idx":    {i},', file=f)
        print('    "num_attack":   1,', file=f)
        print(f'    "eps_init":     {eps},', file=f)
        print('    "max_clique":   3,', file=f)
        print('    "max_search":   1,', file=f)
        print('    "max_level":    1,', file=f)
        print('    "num_classes":  10,', file=f)
        print('    "dp":           1', file=f)
        print('}', file=f)
    #print(subprocess.run([treeVerifyBin, os.path.join(RESULT_DIR, "treeVerify.json")]))



    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    vreal = at.predict_single(instance)
    ax0.plot(range(0, len(bounds)*stepsize, stepsize), [b[1]-b[0] for b in bounds]), 
    ax1.imshow(instance.reshape((28, 28)), cmap="binary")
    ax1.set_title(f"{vreal:.4f}")
    if opt.num_solutions() > 0:
        solutions = opt.solutions()
        #print(solutions)
        instance1 = get_closest_instance(instance, solutions[0][2])
        vfake = at.predict_single(instance1)
        diff = min(instance1 - instance), max(instance1 - instance)
        print("diff", diff)

        print("solution:   ", solutions[0][0:2])
        print("predictions:", vreal, vfake, "(", solutions[0][0], ")")
        ax2.imshow(instance1.reshape((28, 28)), cmap="binary")
        ax2.set_title(f"{vfake:.4f}")
        im = ax3.imshow((instance1-instance).reshape((28, 28)))
        fig.colorbar(im, ax=ax3)

        print("xgb:", model.predict(xgb.DMatrix(np.array([instance, instance1])), output_margin="margin"))
    plt.show()

