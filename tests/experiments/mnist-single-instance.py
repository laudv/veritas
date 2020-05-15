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
    skds.dump_svmlight_file(X[Itest]+1, y[Itest], "tests/data/mnist.libsvm")


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

def set_smt(opt, instance, eps): # for minimization specifically
    smt = io.StringIO()
    for feat_id in opt.get_used_feat_ids()[0]:
        x = opt.xvar(0, feat_id)
        v = instance[feat_id]
        print(f"(assert (<= {x} {v+eps}))", file=smt)
        print(f"(assert (>= {x} {v-eps}))", file=smt)
    for feat_id in opt.get_used_feat_ids()[1]:
        x = opt.xvar(1, feat_id)
        v = instance[feat_id]
        print(f"(assert (<= {x} {v+eps}))", file=smt)
        print(f"(assert (>= {x} {v-eps}))", file=smt)
    opt.set_smt_program(smt.getvalue())

def adv_example_for_eps(instance, label, at0, at1, eps,
        merges,
        opt_stepsize,
        max_opt_time,
        max_opt_cliques):
    opt = Optimizer(at0, at1, set(), False) # share all variables
    set_smt(opt, instance, eps=eps)

    # pruning
    bound_before = opt.current_bounds()
    num_vertices_before = opt.num_vertices(0), opt.num_vertices(1)
    start = timeit.default_timer()
    opt.prune()
    stop = timeit.default_timer()
    num_vertices_after = opt.num_vertices(0), opt.num_vertices(1)
    bound_after = opt.current_bounds()
    opt.reset_smt_program()
    prune_time = stop - start
    #print(f"prune: num_vertices    {num_vertices_before[0]:8} -> {num_vertices_after[0]},",
    #                             f"{num_vertices_before[1]:8} -> {num_vertices_after[1]}")
    #print(f"       bound           {bound_before[0]:8.3f} -> {bound_after[0]:.3f},",
    #                             f"{bound_before[1]:8.3f} -> {bound_after[1]:.3f}")
    #print(f"prune: time            {prune_time:.3f}")

    # merging
    bound_before = opt.current_bounds()
    num_vertices_before = opt.num_vertices(0), opt.num_vertices(1)
    num_indepset_before = opt.num_independent_sets(0), opt.num_independent_sets(1)
    start = timeit.default_timer()
    for m in merges:
        opt.merge(m, instance=0)
        opt.merge(m, instance=1)
    stop = timeit.default_timer()
    num_vertices_after = opt.num_vertices(0), opt.num_vertices(1)
    num_indepset_after = opt.num_independent_sets(0), opt.num_independent_sets(1)
    merge_time = stop - start
    #print(f"prune: num_vertices    {num_vertices_before[0]:8} -> {num_vertices_after[0]},",
    #                             f"{num_vertices_before[1]:8} -> {num_vertices_after[1]}")
    #print(f"       num_indepsets   {num_indepset_before[0]:8} -> {num_indepset_after[0]},",
    #                             f"{num_indepset_before[1]:8} -> {num_indepset_after[1]}")
    #print(f"       bound           {bound_before[0]:8.3f} -> {bound_after[0]:.3f},",
    #                             f"{bound_before[1]:8.3f} -> {bound_after[1]:.3f}")
    #print(f"merge: time            {merge_time:.3f}")

    # optimizing
    bounds = [opt.current_bounds()]
    start = timeit.default_timer()
    if opt_stepsize > 0:
        while opt.num_solutions() == 0 and opt.num_candidate_cliques() < max_opt_cliques:
            if not opt.step(opt_stepsize, 0.0):
                #print("NO SOLUTION")
                break
            b = opt.current_bounds()
            #print(b, opt.nsteps(), opt.num_candidate_cliques())
            bounds.append(b)
            if timeit.default_timer() - start > max_opt_time:
                break
    stop = timeit.default_timer()
    solutions = opt.solutions()
    opt_time = stop - start
    print(f"opt:  step/clique/sol  {sum(opt.nsteps())}/{opt.num_candidate_cliques()}/{len(solutions)}")
    if len(solutions) > 0:
        print(f"       bound           {bound_before[0]:.3f}/{bounds[0][0]:.3f}/{solutions[0][0]:.3f},",
                                     f"{bound_before[1]:.3f}/{bounds[0][1]:.3f}/{solutions[0][1]:.3f}")
    else:
        print(f"       bound           {bound_before[0]:.3f}/{bounds[0][0]:.3f}/{bounds[-1][0]:.3f},",
                                     f"{bound_before[1]:.3f}/{bounds[0][1]:.3f}/{bounds[-1][1]:.3f}")
        print(f"       time            {prune_time:.3f}+{merge_time:.3f}+{opt_time:.3f}")

    # return new lower bound for epsilon if we found one
    if len(solutions) > 0:
        instance1 = get_closest_instance(instance, solutions[0][2])
        print("values:", [x for x in abs(instance1-instance) if x > 0.0])
        print("xgb:", model.predict(xgb.DMatrix(np.array([instance, instance1])), output_margin="margin"))
        eps1 = max(abs(instance1 - instance))
        print(f"       eps             {eps:} -> {eps1}")
        return False, eps1
    else:
        return bounds[-1][1] < bounds[-1][0], eps # it is not possible for at1 to be more certain than at0

def binary_search(nsteps, instance, label, at0, at1, eps,
        merges=[2, 2],
        opt_stepsize=1000,
        max_opt_time=0.5,
        max_opt_cliques=200000):

    verified_eps = 0
    upper = eps
    lower = 0.0

    for step in range(nsteps):
        #print(">> eps =", eps, "step=", step)
        is_unsat, eps1 = adv_example_for_eps(instance, label, at0, at1, eps,
                merges,
                opt_stepsize, max_opt_time, max_opt_cliques)
        print(f" =>  {step:2}: eps = {eps:.3f},", "unsat" if is_unsat else "sat" if eps1!=eps else "maybe")

        old_eps = eps

        # try larger eps
        if is_unsat: # no adv example exists for eps
            verified_eps = eps
            if eps == upper:
                lower = eps
                eps = 2.0 * eps
                upper = eps
            else:
                lower = eps
                eps = eps + 0.5 * (upper - eps)
        else: # we found an adversarial example, or we could not prove that one does not exist
            upper = eps1
            eps = eps - 0.5 * (eps1 - lower)

        print(f" => eps update {old_eps} -> {eps} ({lower}, {upper})")

    return verified_eps




eps = 10
n_adv = 10
instances = X[Itest[0:n_adv]]
labels = y[Itest[0:n_adv]]

for i, (instance, label) in enumerate(zip(instances, labels)):
    at = ats[label]
    print("===================")
    print()

    min_eps = eps
    for j in range(len(ats)):
        if j == label: continue
        print(f"MNIST digit {label} vs. {j} (instance {i})")
        eps0 = binary_search(10, instance, label, at, ats[j], eps, opt_stepsize=0)
        min_eps = min(eps0, min_eps)
        print(f"=> result: {label} vs. {j}: {eps0} [min_eps={min_eps}]")
    
    ##opt = Optimizer(at, ats[0], set(), False)
    #opt = Optimizer(at, minimize=True)
    #eps = 20
    #set_smt(opt, instance, eps=eps)
    #
    #bound_before = opt.current_bounds()
    #num_vertices_before = opt.num_vertices(0)
    #num_vertices_before1 = opt.num_vertices(1)
    #start = timeit.default_timer()
    #opt.prune()
    ##print("time prune:", timeit.default_timer()-start)
    #num_vertices_after = opt.num_vertices(0)
    #num_vertices_after1 = opt.num_vertices(1)
    #opt.reset_smt_program()
    #bound_after = opt.current_bounds()
    #print(f"prune: num_vertices    {num_vertices_before:8} -> {num_vertices_after}")
    #print(f"       bound           {bound_before[0]:8.3f} -> {bound_after[0]:.3f}")
    #print(f"       time            {timeit.default_timer() - start:8.3f}")
    ##print("\n")

    #num_vertices_before = opt.num_vertices(0)
    #num_indep_sets_before = opt.num_independent_sets(0)
    #bound_before = opt.current_bounds()
    #start = timeit.default_timer()
    #opt.merge(2, instance=0)
    #opt.merge(2, instance=1)
    #opt.merge(2, instance=0)
    #opt.merge(2, instance=1) # like in paper
    #num_vertices_after = opt.num_vertices(0)
    #num_indep_sets_after = opt.num_independent_sets(0)
    #bound_after = opt.current_bounds()
    #print(f"merge: num_vertices    {num_vertices_before:8} -> {num_vertices_after}")
    #print(f"       num_indep_sets  {num_indep_sets_before:8} -> {num_indep_sets_after}")
    #print(f"       bound           {bound_before[0]:8.3f} -> {bound_after[0]:.3f}")
    #print(f"       time            {timeit.default_timer() - start:8.3f}")
    ##print("\n\n\n")

    #bounds = [opt.current_bounds()]
    #stepsize = 100

    #start = timeit.default_timer()
    #while opt.num_solutions() == 0 and opt.num_candidate_cliques() < 200000:
    #    #if not opt.step(stepsize, 0.0, 0.0):
    #    if not opt.step(stepsize, 0.0):
    #        #print("NO SOLUTION")
    #        break
    #    b = opt.current_bounds()
    #    #print(b, opt.nsteps(), opt.num_candidate_cliques())
    #    bounds.append(b)
    #print(f"ours:  num_cliques     {opt.num_candidate_cliques():8}")
    #print(f"       num_solutions   {opt.num_solutions():8}")
    #print(f"       bound           {bounds[0][0]:8.3f} -> {bounds[-1][0]:.3f}",
    #        "(optimal)" if opt.num_solutions() > 0 else "",
    #        "(unsat)" if bounds[-1][0] > 0.0 else "")
    #print(f"       time            {timeit.default_timer() - start:8.3f}")

    #print(f"\n\n({bounds[-1][0]:.3f})")
    #print(f"({timeit.default_timer() - start:.3f})")

    ## try treeVerify, assume installed in parent directory
    #treeVerifyBin = "../treeVerification/treeVerify"
    #with open(os.path.join(RESULT_DIR, "treeVerify.json"), "w") as f:
    #    print('{', file=f)
    #    print('    "inputs":       "tests/data/mnist.libsvm", ', file=f)
    #    print(f'    "model":        "{os.path.join(RESULT_DIR, "model.json")}",', file=f)
    #    print(f'    "start_idx":    {i},', file=f)
    #    print('    "num_attack":   1,', file=f)
    #    print(f'    "eps_init":     {eps},', file=f)
    #    print('    "max_clique":   3,', file=f)
    #    print('    "max_search":   1,', file=f)
    #    print('    "max_level":    1,', file=f)
    #    print('    "num_classes":  10,', file=f)
    #    print('    "dp":           1', file=f)
    #    print('}', file=f)
    ##print(subprocess.run([treeVerifyBin, os.path.join(RESULT_DIR, "treeVerify.json")]))



    #fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    #vreal = at.predict_single(instance)
    #ax0.plot(range(0, len(bounds)*stepsize, stepsize), [b[1]-b[0] for b in bounds]), 
    #ax1.imshow(instance.reshape((28, 28)), cmap="binary")
    #ax1.set_title(f"{vreal:.4f}")
    #if opt.num_solutions() > 0:
    #    solutions = opt.solutions()
    #    #print(solutions)
    #    instance1 = get_closest_instance(instance, solutions[0][2])
    #    vfake = at.predict_single(instance1)
    #    diff = min(instance1 - instance), max(instance1 - instance)
    #    print("diff", diff)

    #    print("solution:   ", solutions[0][0:2])
    #    print("predictions:", vreal, vfake, "(", solutions[0][0], ")")
    #    ax2.imshow(instance1.reshape((28, 28)), cmap="binary")
    #    ax2.set_title(f"{vfake:.4f}")
    #    im = ax3.imshow((instance1-instance).reshape((28, 28)))
    #    fig.colorbar(im, ax=ax3)

    #    print("xgb:", model.predict(xgb.DMatrix(np.array([instance, instance1])), output_margin="margin"))
    #plt.show()

