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

def plot_output2(f):
    with open(f) as fh:
        o = json.load(fh)
    #o = o[5:]


    for oo in o:
        b0 = oo["astar"]["bounds"]
        s1 = oo["arastar"]["solutions"]
        e1 = oo["arastar"]["epses"]
        u1 = util.get_ara_bound(e1, s1, task="maximize")[2]
        b2 = oo["merge"]["bounds"]
        t0 = oo["astar"]["timings"]
        t1 = oo["arastar"]["timings"]
        t2 = oo["merge"]["timings"]

        print("epses:", e1)
        print("merge levels:", len(oo["merge"]["bounds"]))

        fig, ax = plt.subplots(figsize=(12, 7))
        
        if len(oo["astar"]["solutions"]) > 0:
            s = oo["astar"]["solutions"][0]
            ax.axhline(s, color="gray", linestyle=":", linewidth=1, label="Optimal")
            b0 = [x for x in b0 if x >= s]
            t0 = [y for x, y in zip(b0, t0) if x >= s]
        else:
            ax.axhline(min(b0), color="gray", linestyle=":", linewidth=1, label="A* upper bound")
            ax.axhline(max(s1), color="gray", linestyle=":", linewidth=1, label="ARA* lower bound")
        ax.plot(t0, b0, label="A*")
        ax.plot(t1, s1, ".-", label="ARA*")
        ax.plot(t1, u1, "-.", label="ARA* upper")
        l, = ax.plot(t2, b2, "x-", label="merge")
        if True or len(oo["astar"]["solutions"]) == 0: # should actually be disabled when merge finds the solution
            #t2.append(max(t0[-1], t1[-1]))
            #b2.append(b2[-1])
            t3 = max(t0[-1], t1[-1])
            ax.plot([t2[-1], t3], [b2[-1], b2[-1]], "-", color=l.get_color())
            ax.text(t3, b2[-1], "OOT", horizontalalignment='right', verticalalignment='bottom', color=l.get_color())
        ax.set_title(f"num_trees = {oo['num_trees']}")
        ax.legend()
        ax.set_xlabel("time");
        ax.set_ylabel("model output");
        plt.show()


if __name__ == "__main__":
    #plot_output("tests/experiments/scale/output5G")
    plot_output2("tests/experiments/scale/output5G")


