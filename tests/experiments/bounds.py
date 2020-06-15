import timeit
import matplotlib.pyplot as plt 
import numpy as np
from treeck import *

def prune_calhouse(opt, instance):
    # median housing age > 50, median income > 10
    print("num_vertices before:", opt.num_vertices(instance))
    opt.enable_smt()
    opt.set_smt_program(f"""
(assert (> {opt.xvar(instance, 2)} 50.0))
(assert (> {opt.xvar(instance, 7)} 10.0))""")
    opt.prune()
    opt.disable_smt()
    print("num_vertices after:", opt.num_vertices(instance))

def test_calhouse():
    nsteps = 100
    max_candidates = 1000000
    instance = 1 # 0 = minimize, 1 = maximize

    # ==== A*
    at = AddTree.read("tests/models/xgb-calhouse-intermediate.json")
    opt = Optimizer(maximize=at)
    prune_calhouse(opt, instance)
    opt.set_ara_eps(1.0, 0.0)
    
    start = timeit.default_timer()
    current_bounds = [opt.current_bounds()]
    timings1 = [0.0]
    while opt.num_solutions() == 0 and opt.num_candidate_cliques() < max_candidates:
        if not opt.step(nsteps):
            break
        current_bounds.append(opt.current_bounds())
        timings1.append(timeit.default_timer() - start)

    opt_solutions = opt.solutions()
    print("A*: ", [s[instance] for s in opt.solutions()])


    # ==== ARA*
    opt = Optimizer(maximize=at)
    prune_calhouse(opt, instance)
    opt.set_ara_eps(0.05, 0.05)
    
    start = timeit.default_timer()
    timings2 = []
    epses = []
    while opt.num_candidate_cliques() < max_candidates and opt.get_ara_eps() < 1.0:
        if not opt.step(nsteps):
            break
        while len(timings2) < opt.num_solutions():
            timings2.append(timeit.default_timer() - start)
            epses.append(opt.get_ara_eps())

    ara_solutions = opt.solutions()
    end = np.argmax([s[instance] for s in ara_solutions]) + 1
    print("ARA*: ", ara_solutions[end][instance])


    # ==== Merging
    opt = Optimizer(maximize=at, max_memory=1024*1024*256)
    prune_calhouse(opt, instance)

    start = timeit.default_timer()
    timings3 = [0.0]
    merge_bounds = [opt.current_bounds()]
    print("merge:", opt.num_vertices(instance))
    while opt.num_vertices(instance) < max_candidates:
        opt.merge(2)
        if merge_bounds[-1][instance] == opt.current_bounds()[instance]:
            break
        print("merge:", opt.num_vertices(instance), opt.current_bounds()[instance])
        merge_bounds.append(opt.current_bounds())
        timings3.append(timeit.default_timer() - start)


    # Plot results
    fig, (ax, ax2) = plt.subplots(1, 2)
    l1, = ax.plot(timings1, [x[instance] for x in current_bounds], label="A*")
    l2, = ax.plot(timings2, [s[instance] for s in ara_solutions], ".", label="ARA*")
    l3, = ax.plot(timings3, [x[instance] for x in merge_bounds], ".--", label="NeurIPS")
    if len(opt_solutions) > 0:
        ax.axhline(opt_solutions[0][instance], ls=":", lw=1, c="gray", label="Optimal")
    else:
        ax.axhline(current_bounds[-1][instance], ls=":", lw=1, c=l1.get_color(), label="A* best (upper)")
        ax.axhline(ara_solutions[end-1][instance], ls=":", lw=1, c=l2.get_color(), label="ARA* best (lower)")

    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("model output")
    ax.set_title("Ensemble output bounds")

    ax2.plot(timings2, epses, c="gray", label="ARA* eps")
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("eps")
    ax2.set_title("f(c) = g(c) + eps * h(c)")
    plt.show()

test_calhouse()
