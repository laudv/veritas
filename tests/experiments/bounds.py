import timeit
import matplotlib.pyplot as plt 
import numpy as np
from treeck import *

def test_calhouse():
    nsteps = 1000
    max_candidates = 10000000
    instance = 1 # 0 = minimize, 1 = maximize

    at = AddTree.read("tests/models/xgb-calhouse-hard.json")
    opt = Optimizer(maximize=at)
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

    opt = Optimizer(maximize=at)
    opt.set_ara_eps(0.02, 0.02)
    
    start = timeit.default_timer()
    timings2 = []
    epses = []
    while opt.num_candidate_cliques() < max_candidates and opt.get_ara_eps() < 1.0:
        if not opt.step(nsteps):
            break
        while len(timings2) < opt.num_solutions():
            timings2.append(timeit.default_timer() - start)
            epses.append(opt.get_ara_eps())

    end = np.argmax([s[instance] for s in opt.solutions()]) + 1
    print("ARA*: ", [s[instance] for s in opt.solutions()][:end])

    fig, ax = plt.subplots()
    ax.plot(timings1, [x[instance] for x in current_bounds], label="A*")
    ax.plot(timings2[:end], [s[instance] for s in opt.solutions()][:end], "x", label="ARA*")
    #ax.plot(timings2[:end], epses[:end], ":", c="gray", label="ARA* eps")
    if len(opt_solutions) > 0:
        ax.axhline(opt_solutions[0][instance], ls=":", lw=1, c="gray", label="Optimal")

    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("model output")
    plt.show()

#test_calhouse()
