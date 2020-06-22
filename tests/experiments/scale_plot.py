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
        b2 = oo["merge"]["bounds"]
        t0 = oo["astar"]["timings"]
        t1 = oo["arastar"]["timings"]
        t2 = oo["merge"]["timings"]

        # if the next solution is worse, then its better eps transfers to the previous solution
        e1f = e1.copy()
        change = True
        while change:
            change = False
            for i in range(0, len(e1f)-1):
                if s1[i] > s1[i+1] and e1f[i] < e1f[i+1]: # better solution, but worse eps
                    change = True
                    print(f"e1f: {i}: {e1f[i]} -> {e1f[i+1]} [{s1[i]}, {s1[i+1]}]")
                    e1f[i] = e1f[i+1]

        # for the ara* upper bound, always use the current best solution
        s1b = s1.copy()
        for i in range(1, len(s1)):
            if s1b[i] < s1b[i-1]:
                print(f"s1b: {i} [{s1b[i-1]}, {s1b[i]}]")
                s1b[i] = s1b[i-1]

        print("epses:", oo["arastar"]["epses"])
        print("levels:", len(oo["merge"]["bounds"]))

        fig, ax = plt.subplots()
        
        print(len(t0), len(b0))
        ax.plot(t0, b0, label="A*")
        if len(oo["astar"]["solutions"]) > 0:
            ax.axhline(oo["astar"]["solutions"][0], color="gray", linestyle=":", linewidth=1, label="A* best")
        print(len(t1), len(s1))
        ax.plot(t1, s1, "o-", label="ARA*")
        ax.plot(t1, [s/e for s, e in zip(s1b, e1f)], label="ARA* upper")
        print(len(t2), len(b2))
        ax.plot(t2, b2, "x-", label="merge")
        ax.plot(t1, util.get_ara_bound(e1, s1, task="maximize")[2], "-.", label="ARA* upper 2")
        ax.set_title(f"num_trees = {oo['num_trees']}")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    #plot_output("tests/experiments/scale/output_10G_merge6_epsinc001")
    plot_output2("tests/experiments/scale/output5G")


