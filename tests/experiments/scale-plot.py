import os, json
import numpy as np
import matplotlib.pyplot as plt
import util

RESULT_DIR = "tests/experiments/scale"

def plot_output(f):
    with open(f) as fh:
        o = json.load(fh)
    #o = o[2:]

    x = [oo["num_trees"] for oo in o]

    m0 = [oo["astar"]["memory"][-1] for oo in o]
    n0 = [oo["astar"]["steps"][-1] for oo in o]
    b0 = [oo["astar"]["bounds"][-1]
            if len(oo["astar"]["solutions"]) == 0
            else oo["astar"]["solutions"][0]
            for oo in o]

    e1 = [oo["arastar"]["epses"][-1] for oo in o]
    s1 = [max(oo["arastar"]["solutions"]) for oo in o]

    b2 = [oo["merge"]["bounds"][-1] for oo in o]

    print("merge levels:", [len(oo["merge"]["bounds"]) for oo in o])

    for oo in o:
        print(oo["arastar"]["epses"])

    fig, ax = plt.subplots()

    ax.plot(x, b0, label="upper bound")
    ax.plot(x, s1, label="lower bound")
    ax.plot(x, [(s / e) for s, e in zip(s1, e1)], "--", label="lower bound bound")
    ax.plot(x, b2, label="merge upper bound")

    #ax.plot(x, [b - s for s, b in zip(s1, b0)], label="bound gap")
    #ax.plot(x, [(s / e) - s for s, e in zip(s1, e1)], "--", label="bound bound gap")
    #ax.plot(x, [b - s for s, b in zip(s1, b2)], label="bound gap merge")

    ax.legend()
    plt.show()

def plot_output2(*args):
    filenames = [os.path.basename(f) for f in args]
    jsons = []
    for f in args:
        with open(f) as fh:
            jsons.append(json.load(fh))

    for oos in zip(*jsons):
        n = len(oos)

        fig, axs = plt.subplots(n, 1, figsize=(8, 5*n), sharey=True)
        axs = list(axs)

        for oo, ax, name in zip(oos, axs, filenames):
            b0 = oo["astar"]["bounds"][1:]
            s0 = oo["astar"]["solutions"]
            s1 = oo["arastar"]["solutions"]
            e1 = oo["arastar"]["epses"]
            s1f, e1f, b1 = util.get_ara_bound(e1, s1, task="maximize")
            b2 = oo["merge"]["bounds"]
            t0 = oo["astar"]["timings"][1:]
            t1 = oo["arastar"]["timings"]
            t2 = oo["merge"]["timings"]

            print("== num_trees", oo["num_trees"], name)
            print("best eps:", e1[-1])
            print("best solution ARA*", max(s1), f"({len(s1)})")
            print("best solution A*  ", s0[0] if len(s0) > 0 else "NA", f"({len(s0)})")
            print("best bound A*     ", min(b0))
            print("max memory        ", max(oo["astar"]["memory"])/(1024*1024))
            print("merge levels:", len(oo["merge"]["bounds"]))

            if len(s0) > 0:
                ax.axhline(s0[0], color="gray", linestyle=":", linewidth=1, label="Optimal")
                b0.append(s0[0])
                t0.append(t0[-1])
                b0 = [x for x in b0 if x >= s0[0]]
                t0 = [y for x, y in zip(b0, t0) if x >= s0[0]]
            else:
                ax.axhline(min(b0), color="gray", linestyle=":", linewidth=1, label="A* upper bound")
                ax.axhline(max(s1), color="gray", linestyle=":", linewidth=1, label="ARA* lower bound")
            ax.plot(t0, b0, label="A*")
            ax.plot(t1, s1f, "-", label="ARA*")
            ax.plot(t1, b1, "-.", label="ARA* upper")
            l, = ax.plot(t2, b2, "x-", label="merge")
            if True or len(oo["astar"]["solutions"]) == 0: # should actually be disabled when merge finds the solution
                #t2.append(max(t0[-1], t1[-1]))
                #b2.append(b2[-1])
                t3 = max(t0[-1], t1[-1])
                if t3 > t2[-1]:
                    ax.plot([t2[-1], t3], [b2[-1], b2[-1]], "-", color=l.get_color())
                    ax.text(t3, b2[-1], "OOT", horizontalalignment='right', verticalalignment='bottom', color=l.get_color())
            ax.set_title(f"num_trees = {oo['num_trees']} ({name})")
            ax.legend()
            ax.set_xlabel("time");
            ax.set_ylabel("model output");

        plt.show()


if __name__ == "__main__":
    #plot_output("tests/experiments/scale/output5G")
    plot_output2("tests/experiments/scale-covtype/neww5G_merge0", "tests/experiments/scale-covtype/new5G_merge0")
    #plot_output2("tests/experiments/scale-mnist/output2_dp", "tests/experiments/scale-mnist/output2")


