import sys, os, json, glob, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import util

#import seaborn as sns

RESULT_DIR = "tests/experiments/scale"

def plot_output1(*args):
    filenames = [os.path.basename(f) for f in args]
    jsons = []
    for f in args:
        if f.endswith(".gz"):
            with gzip.open(f, "r") as fh:
                jsons.append(json.load(fh))
        else:
            with open(f) as fh:
                jsons.append(json.load(fh))

    for oos in zip(*jsons):
        n = len(oos)

        fig, axs = plt.subplots(n, 1, figsize=(8, 5*n), sharey=True, sharex=True)

        try: axs[0]
        except:
            axs = np.array([axs])

        oot_pos = 0
        for oo, ax, name in zip(oos, axs, filenames):
            print(f"\n== {name}: num_trees {oo['num_trees']}, depth {oo['depth']} ==")

            # A*
            tb0 = oo["a*"]["bounds_times"]
            b0  = [x[1]-x[0] for x in oo["a*"]["bounds"]]
            ts0 = oo["a*"]["sol_times"]
            s0  = [x[1]-x[0] for x in oo["a*"]["solutions"]]

            if len(s0) > 0:
                print("A* optimal:", s0[0])
                ax.axhline(s0[0], color="gray", linestyle=(0, (2, 4)), linewidth=1, label="Solution")
                b0.append(s0[0])
                tb0.append(ts0[0])
                b0 = [x for x in b0 if x >= s0[0]]
                tb0 = [y for x, y in zip(b0, tb0) if x >= s0[0]]
            else:
                print("A* best:", min(b0))
                ax.axhline(min(b0), color="gray", linestyle=(0, (2, 4)), linewidth=1, label="A* best")
            ax.plot(tb0, b0, label="A* upper")
            if "best_solution_box" in oo["a*"]:
                print("A* sol: ", oo["a*"]["best_solution_box"])

            # ARA*
            tb1 = oo["ara*"]["bounds_times"]
            if len(oo["ara*"]["solutions"]) > 0:
                b1  = [x[1]-x[0] for x in oo["ara*"]["bounds"]]
                ts1 = oo["ara*"]["sol_times"]
                s  = [x[1]-x[0] for x in oo["ara*"]["solutions"]]
                s10  = [x[0] for x in oo["ara*"]["solutions"]]
                s11  = [x[1] for x in oo["ara*"]["solutions"]]
                e1  = oo["ara*"]["epses"]
                d1 = oo["ara*"]["total_time"]
                #s10f, s11f, ts1f, e1f = util.filter_solutions(s10, s11, ts1, e1)
                #s1f = [b-a for a, b in zip(s10f, s11f)]
                #b1f = util.flatten_ara_upper(s1f, e1f)

                l1, = ax.plot(ts1, s, ".-", label="ARA* lower")
                #ax.plot(ts1f, b1f, label="ARA* upper", ls=(0, (2, 2)), c=l1.get_color())
                #ylim_lo, ylim_hi = ax.get_ylim()
                #ax.plot(tb1, b1, ".", markersize=1.5, c=l1.get_color())
                #ax.set_ylim(bottom=ylim_lo)
                print("ARA* best:", max(s), "eps:", max(e1))
                if "best_solution_box" in oo["ara*"]:
                    print("ARA* sol: ", oo["ara*"]["best_solution_box"])
                if len(s0) == 0:
                    ax.axhline(max(s), color="gray", ls=(4, (2, 4)), lw=1, label="ARA* best")

            # merge
            if "merge" in oo:
                b2 = [x[1][1]-x[0][0] for x in oo["merge"]["bounds"]]
                t2 = oo["merge"]["times"]
                oot = oo["merge"]["oot"]
                oom = oo["merge"]["oom"]
                tt = oo["merge"]["total_time"]
                mt = oo["max_time"]
                mm = oo["max_memory"]
                l2, = ax.plot(t2, b2, "x-", label="Merge")
                if oot or oom:
                    label = f"OOM ({mm/(1024*1024*1024):.1f}gb, {tt:.0f}s)" if oom else f"OOT ({mt}s)"
                    oot_pos = max(oot_pos, max(tb0), max(tb1), max(t2))
                    ax.plot([t2[-1], oot_pos], [b2[-1], b2[-1]], ":", color=l2.get_color())
                    ax.text(oot_pos, b2[-1], label, horizontalalignment='right',
                            verticalalignment='bottom', color=l2.get_color())

                print("merge best:", min(b2), "OOT:", oot, "OOM:", oom, "optimal", oo["merge"]["optimal"])

            # plot details
            ax.set_title(f"num_trees={oo['num_trees']}, depth={oo['depth']} ({name})")
            ax.legend()
            ax.set_xlabel("time");
            ax.set_ylabel("model output");
            ax.set_ylim(top=1.1*max(b0));
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

        plt.show()

def plot_output2(f, i):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), sharey=True, sharex=True)
    with open(f) as fh:
        oo = json.load(fh)[i]

    print(list(oo["a*"].keys()))

    # A*
    tb0 = oo["a*"]["bounds_times"]
    b0  = [x[1]-x[0] for x in oo["a*"]["bounds"]]
    ts0 = oo["a*"]["sol_times"]
    s0  = [x[1]-x[0] for x in oo["a*"]["solutions"]]

    if len(s0) > 0:
        print("A* optimal:", s0[0])
        ax.axhline(s0[0], color="gray", linestyle=(0, (2, 4)), linewidth=1, label="Solution")
        b0.append(s0[0])
        tb0.append(ts0[0])
        b0 = [x for x in b0 if x >= s0[0]]
        tb0 = [y for x, y in zip(b0, tb0) if x >= s0[0]]
    else:
        print("A* best:", min(b0))
        ax.axhline(min(b0), color="gray", linestyle=(0, (2, 4)), linewidth=1, label="A* best")
    ax.plot(tb0, b0, label="A* upper")
    if "best_solution_box" in oo["a*"]:
        print("A* sol: ", oo["a*"]["best_solution_box"])

    # ARA*
    tb1 = oo["ara*"]["bounds_times"]
    if len(oo["ara*"]["solutions"]) > 0:
        b1  = [x[1]-x[0] for x in oo["ara*"]["bounds"]]
        ts1 = oo["ara*"]["sol_times"]
        s10  = [x[0] for x in oo["ara*"]["solutions"]]
        s11  = [x[1] for x in oo["ara*"]["solutions"]]
        e1  = oo["ara*"]["epses"]
        d1 = oo["ara*"]["total_time"]
        s10f, s11f, ts1f, e1f = util.filter_solutions(s10, s11, ts1, e1)
        s1f = [b-a for a, b in zip(s10f, s11f)]
        b1f = util.flatten_ara_upper(s1f, e1f)

        l1, = ax.plot(ts1f, s1f, ".-", label="ARA* lower")
        ax.plot(ts1f, b1f, label="ARA* upper", ls=(0, (2, 2)), c=l1.get_color())
        #ylim_lo, ylim_hi = ax.get_ylim()
        #ax.plot(tb1, b1, ".", markersize=1.5, c=l1.get_color())
        #ax.set_ylim(bottom=ylim_lo)
        print("ARA* best:", max(s1f), "eps:", max(e1))
        if "best_solution_box" in oo["ara*"]:
            print("ARA* sol: ", oo["ara*"]["best_solution_box"])
        if len(s0) == 0:
            ax.axhline(max(s1f), color="gray", ls=(4, (2, 4)), lw=1, label="ARA* best")

    # merge
    if "merge" in oo:
        b2 = [x[1]-x[0] for x in oo["merge"]["bounds"]]
        t2 = oo["merge"]["times"]
        oot = oo["merge"]["oot"]
        oom = oo["merge"]["oom"]
        tt = oo["merge"]["total_time"]
        mt = oo["max_time"]
        mm = oo["max_memory"]
        l2, = ax.plot(t2, b2, "x-", label="Merge")
        oot_pos = 0
        if oot or oom:
            label = f"OOM ({mm/(1024*1024*1024):.1f}gb, {tt:.0f}s)" if oom else f"OOT ({mt}s)"
            oot_pos = max(oot_pos, max(tb0), max(tb1), max(t2))
            ax.plot([t2[-1], oot_pos], [b2[-1], b2[-1]], ":", color=l2.get_color())
            ax.text(oot_pos, b2[-1], label, horizontalalignment='right',
                    verticalalignment='bottom', color=l2.get_color())

        print("merge best:", min(b2), "OOT:", oot, "OOM:", oom, "optimal", oo["merge"]["optimal"])

    # plot details
    ax.set_title(f"num_trees={oo['num_trees']}, depth={oo['depth']} ({os.path.basename(f)})")
    ax.legend()
    ax.set_xlabel("time");
    ax.set_ylabel("model output");
    #ax.set_ylim(bottom=min(b1), top=1.1*max(b0));
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    plt.show()

def plot_output3(file):
    with open(file) as fh:
        oo = json.load(fh)

    depths = {}
    for depth in [3,4,6,8]:#range(4, 9, 2):
        ooo = [o for o in oo if o["depth"] == depth]
        if len(oo) == 0: continue
        depths[depth] = ooo

    fig, axs = plt.subplots(1, len(depths), sharey=True)#, figsize=(4, 2.5))

    for i, ((depth, oo), ax) in enumerate(zip(depths.items(), axs)):
        xs = [o["num_trees"] for o in oo]
        A = [util.get_best_astar(o["a*"]) for o in oo]
        ARA = [max(map(lambda b: b[1], o["ara*"]["solutions"]))
                if len(o["ara*"]["solutions"]) > 0 else -np.inf
                for o in oo]
        ARAeps = [o["ara*"]["epses"][-1]
                if len(o["ara*"]["epses"]) > 0 else 0.0
                for o in oo]
        mergelo = [o["merge"]["bounds"][-1][1][0] for o in oo]
        mergehi = [o["merge"]["bounds"][-1][1][1] for o in oo]

        relA = [1.0 for a in A]
        relARA = [ara/a for a, ara in zip(A, ARA)]
        relmlo = [m/a for a, m in zip(A, mergelo)]
        relmhi = [m/a for a, m in zip(A, mergehi)]

        #ax.fill_between(x, relA, relARA, color="lightgray")
        #ax.semilogx(x, relA, label="A*")
        #ax.semilogx(x, relARA, label="ARA*")
        #ax.semilogx(x, relmerge, label="merge")
        #ax.semilogx(x, ARAeps, label="ARA* eps", color="black", ls="--")
        #ax.semilogx(x, A, label="A")
        #ax.semilogx(x, ARA, label="ARA")
        #ax.semilogx(x, mergelo, label="merge lower")
        #ax.semilogx(x, mergehi, label="merge higher")

        xxs = np.arange(len(xs))# + i/len(depths)
        #ours = ax.bar(xx,
        #        [hi-lo for lo, hi in zip(ARA, A)], bottom=ARA,
        #        zorder=9, width=0.8/len(depths), color="blue")
        #theirs = ax.bar(xx,
        #        [hi-lo for lo, hi in zip(mergelo, mergehi)], bottom=mergelo,
        #        zorder=1, width=0.8/len(depths), color="red")
        #ours = ax.bar(xx,
        #        [hi-lo for lo, hi in zip(relARA, relA)], bottom=relARA,
        #        zorder=9, align="edge", width=0.8/len(depths), color="blue")
        #theirs = ax.bar(xx,
        #        [hi-lo for lo, hi in zip(relmlo, relmhi)], bottom=relmlo,
        #        zorder=1, align="edge", width=0.8/len(depths), color="red")

        def interval(ax, x, lo, hi, lw=1, label=None):
            ax.vlines(x, lo+0.1, hi-0.1, lw=lw)
            ax.hlines(lo+0.1, x-0.2, x+0.2, lw=lw)
            ax.hlines(hi-0.1, x-0.2, x+0.2, lw=lw)
            #if label is not None:
            #    ax.text(x, hi, label, horizontalalignment="center")

        def interval_dashed(ax, x, lo, hi, lw=1):
            ax.vlines(x, lo, hi, lw=lw, linestyles ="dashed")
            ax.hlines(lo, x-0.1, x+0.1, lw=lw)
            ax.hlines(hi, x-0.1, x+0.1, lw=lw)

        for x, lo, hi in zip(xxs, ARA, A):
            interval(ax, x, lo, hi, lw=2.0)

        for x, lo, hi in zip(xxs, mergelo, mergehi):
            interval_dashed(ax, x, lo, hi, lw=1.0)

        ax.set_xlabel("#trees")
        ax.set_title(f"depth {depth}")

        ax.set_xticks(xxs)
        ax.set_xticks(xxs, minor=True)
        ax.set_xticklabels([str(s) for s in xs], rotation=90)

    axs[0].set_ylabel("model output")
    axs[0].legend([
            Line2D([0], [0], color="black", lw=2),
            Line2D([0], [0], ls="--", color="black", lw=1)
        ], ["ours", "merge"],
        bbox_to_anchor=(0.2, 1.15, 4.3, 0.0), loc='lower left', ncol=2,
        mode="expand", borderaxespad=0.0, frameon=False)
    #plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.15, right=0.95)
    plt.show()

def plot_output4(file, depth):
    fig, ax = plt.subplots(1, 1)#, figsize=(4, 2.5))
    with open(file) as fh:
        oo = json.load(fh)
    oo = [o for o in oo if o["depth"] == depth]

    x = [o["num_trees"] for o in oo]
    #A = [max(o["a*"]["cliques"]) for o in oo]
    #ARA = [max(o["ara*"]["cliques"]) for o in oo]
    #merge = [max(o["merge"]["vertices"]) for o in oo]

    A = [max(o["a*"]["memory"]) / (1024*1024) for o in oo]
    ARA = [max(o["ara*"]["memory"]) / (1024*1024) for o in oo]
    merge = [max(o["merge"]["memory"]) / (1024*1024) for o in oo]

    At = [max(o["a*"]["bounds_times"]) for o in oo]
    ARAt = [max(o["ara*"]["bounds_times"]) for o in oo]
    merget = [max(o["merge"]["times"]) for o in oo]
    Apt = [a/t for a, t in zip(A, At)]
    ARApt = [a/t for a, t in zip(ARA, ARAt)]
    mergept = [a/t for a, t in zip(merge, merget)]

    print(Apt)
    print(mergept)

    #ax.semilogx(x, A, label="A*")
    #ax.semilogx(x, ARA, label="ARA*")
    #ax.semilogx(x, merge, label="merge")
    ax.loglog(x, Apt, label="A*")
    ax.loglog(x, ARApt, label="ARA*")
    ax.loglog(x, mergept, label="merge")

    ax.set_xticks(x)
    ax.set_xticks(x, minor=True)
    ax.set_xticklabels([str(s) for s in x])
    
    ax.set_xlabel("#trees")
    ax.set_ylabel("Mb per sec.")
    ax.set_title("memory consumption per second")

    plt.legend()
    plt.show()

def plot_output5(pattern):
    oo = []
    for f in glob.glob(f"tests/experiments/scale/{pattern}"):
        with open(f) as fh:
            oo += json.load(fh)
    print(len(oo), "records")
    
    fig, ax = plt.subplots(1, 1)#, figsize=(4, 2.5))

    print(list(oo[0]["a*"].keys()))
    print(list(oo[0]["ara*"].keys()))

    num_vertices = [o["a*"]["num_vertices0"] + o["a*"]["num_vertices1"] for o in oo]
    A = [util.get_best_astar(o["a*"]) for o in oo]
    ARA = [max(map(lambda b: b[1]-b[0], o["ara*"]["solutions"]))
            if len(o["ara*"]["solutions"]) > 0
            else -np.inf
            for o in oo]
    merge = [min(map(lambda b: b[1], o["merge"]["bounds"])) for o in oo]

    A, ARA, merge = np.array(A), np.array(ARA), np.array(merge)

    #l0, = ax.plot(num_vertices, [(a-ara)/a for a, ara in zip(A, ARA)], ".", alpha=0.2, zorder=-1, markersize=5)
    l1, = ax.plot(num_vertices, [(m-a) for a, m in zip(A, merge)], ".", alpha=0.5, zorder=-1, markersize=5)
    #l0, = ax.plot(num_vertices, [a-ara for a, ara in zip(A, ARA)], ".", alpha=0.05, zorder=-1, markersize=20)
    #l1, = ax.plot(num_vertices, [m-a for a, m in zip(A, merge)], ".", alpha=0.05, zorder=-1, markersize=20)

    #bins = np.linspace(min(num_vertices), max(num_vertices), 20)
    #bin_width = bins[1]-bins[0]
    #assignments = np.digitize(num_vertices, bins)
    #meanA = [np.mean(merge[assignments==int(bin)]-A[assignments==int(bin)]) for bin in range(len(bins))]
    #stdA = [np.std(merge[assignments==int(bin)]-A[assignments==int(bin)]) for bin in range(len(bins))]

    #ax.bar(bins, meanA, 0.45*bin_width, yerr=stdA, color=l1.get_color())

    #for x, a, ara in zip(num_vertices, A, ARA):
    #    ax.plot([x, x], [a, ara], ".-b", alpha=0.25)
    #for x, a, m in zip(num_vertices, A, merge):
    #    ax.plot([x, x], [a, m], "-b", alpha=0.25)

    ax.set_xlabel("number of reachable leafs")
    ax.set_ylabel("(merge - A*) bounds (> 0 means A* wins)")
    ax.set_title("merge vs A*: how do upper bounds compare")
    plt.show()

def time_to_beat_merge(o):
    A = [b[1]-b[0] for b in o["a*"]["bounds"]]
    At = o["a*"]["bounds_times"]
    merge = o["merge"]["bounds"][-1]
    merge = merge[1] - merge[0]

    #print(A)
    #print(At)
    #print(merge)

    try:
        return [at for a, at in zip(A, At) if a < merge][0]
    except:
        return np.inf

def plot_output6(pattern):
    oo = []
    for f in glob.glob(f"tests/experiments/scale/{pattern}"):
        with open(f) as fh:
            oo += json.load(fh)
    print(len(oo), "records")

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,
            gridspec_kw={"height_ratios": [4, 1]})#, figsize=(4, 2.5))

    num_vertices = [o["a*"]["num_vertices0"] + o["a*"]["num_vertices1"] for o in oo]
    At = [util.get_best_astar(o["a*"]) for o in oo]
    mt = [o["merge"]["times"][-1] for o in oo]
    ms = [o["merge"]["vertices"][-1] for o in oo]
    Ab = [time_to_beat_merge(o) for o in oo]

    print("mean time to beat time", np.mean(Ab))

    ratio = np.array([m-a for a, m in zip(Ab, mt)])

    l0, = ax.plot(num_vertices, ratio, ".", alpha=0.5, zorder=20, markersize=5)
    ax.plot(num_vertices, Ab, ".", alpha=0.5, zorder=20, markersize=5)
    ax2.plot(num_vertices, [len(o["merge"]["bounds"]) for o in oo], ".", markersize=5,
            color="gray", zorder=10, alpha=0.5)

    #bins = np.linspace(min(num_vertices), max(num_vertices), 10)
    #bin_width = bins[1]-bins[0]
    #assignments = np.digitize(num_vertices, bins)

    #meanA = [np.mean(ratio[assignments==int(bin)]) for bin in range(len(bins))]
    #meanA = [sum(assignments==int(bin)) for bin in range(len(bins))]
    #stdA = [np.std(ratio[assignments==int(bin)]) for bin in range(len(bins))]

    #for b in range(5, len(bins)):
    #    data = ratio[assignments==int(b)]
    #    print(data)
    #    ax.boxplot(data)
    #    break

    #ax.bar(bins, meanA, 0.45*bin_width)

    #sns.set(style="whitegrid", palette="pastel", color_codes=True)
    #data = pd.DataFrame({"bin": assignments, "value": ratio})
    #sns.violinplot(x="bin", y="value", data=data)
    ax2.set_xlabel("number of reachable leafs")
    ax.set_ylabel("(merge - A*) time (> 0 means A* wins)")
    ax2.set_ylabel("merge level")
    ax2.set_yticks(range(4, 9))
    ax.set_title("merge vs A*: time for A* to get to best bound of merge")
    plt.show()

def plot_robust(pattern):
    cache_file = f"/tmp/temporary_mnist_glob_cache_{pattern}.h5"

    if not os.path.exists(cache_file):
        oo = []
        for f in glob.glob(f"tests/experiments/scale/mnist/{pattern}"):
            with gzip.open(f, "r") as fh:
                oo += json.load(fh)

        print(len(oo))
        print(list(oo[0].keys()))

        data = {}

        A = [util.get_best_astar(o["a*"], task="both") for o in oo]
        data["A0"] = [a[0] for a in A]
        data["A1"] = [a[1] for a in A]
        ARA = [max(o["ara*"]["solutions"], key=lambda b: b[1]-b[0])
                if len(o["ara*"]["solutions"]) > 0
                else (np.inf, -np.inf)
                for o in oo]
        data["ARA0"] = [ara[0] for ara in ARA]
        data["ARA1"] = [ara[1] for ara in ARA]
        data["ARAeps"] = [max(o["ara*"]["epses"])
                if len(o["ara*"]["epses"]) > 0
                else 0.0
                for o in oo]
        merge = [o["merge"]["bounds"][-1] for o in oo]
        data["merge0"] = [m[0] for m in merge]
        data["merge1"] = [m[1] for m in merge]
        data["merge_time"] = [o["merge"]["times"][-1] for o in oo]
        data["time_to_beat_merge"] = [time_to_beat_merge(o) for o in oo]
        data["example_seed"] = [o["example_seed"] for o in oo]
        data["example_i"] = [o["example_i"] for o in oo]
        data["delta"] = [o["delta"] for o in oo]
        data["source"] = [o["example_label"] for o in oo]
        data["target"] = [o["target_label"] for o in oo]
        data["Aissol"] = [len(o["a*"]["solutions"]) > 0 for o in oo]
        data["mergeissol"] = [o["merge"]["bounds"][0]==o["merge"]["bounds"][1] for o in oo]
        data["binsearch_step"] = [o["binsearch_step"] for o in oo]
        data["done_early"] = ["done_early" in o for o in oo]
        data["a_unsat"] = [a[0] > a[1] for a in A]
        data["merge_unsat"] = [m[0] > m[1] for m in merge]
        data["up"] = [o["binsearch_upper"] for o in oo]
        data["lo"] = [o["binsearch_lower"] for o in oo]


        df = pd.DataFrame(data)
        df["a_unsat_merge"] = [a[0]> a[1] and m[0]<=m[1] for a, m in zip(A, merge)]
        df["a_merge_unsat"] = [a[0]<=a[1] and m[0]> m[1] for a, m in zip(A, merge)]

        df.to_hdf(cache_file, key="data")
    else:
        df = pd.read_hdf(cache_file)

    #print(df)


    dfg = df.groupby(["example_seed", "example_i", "source", "target"])
    x = dfg["a_unsat_merge"].any() # was at any point in the search A* better than merge?
    #print(dfg.get_group((2, 11323, 4, 5)))

    for b, i in zip(x, x.index):
        if not b: continue
        print(i)
        g = dfg.get_group(i)
        print(g)
        #break


if __name__ == "__main__":
    if int(sys.argv[1]) == 1:
        plot_output1(sys.argv[2])
    if int(sys.argv[1]) == 2:
        plot_output2(sys.argv[2], int(sys.argv[3]))
    if int(sys.argv[1]) == 3:
        plot_output3(sys.argv[2])
    if int(sys.argv[1]) == 4:
        plot_output4(sys.argv[2], int(sys.argv[3]))
    if int(sys.argv[1]) == 5:
        plot_output5(sys.argv[2])
    if int(sys.argv[1]) == 6:
        plot_output6(sys.argv[2])
    if int(sys.argv[1]) == 7:
        plot_robust(sys.argv[2])
