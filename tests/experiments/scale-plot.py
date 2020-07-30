import sys, os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

#import seaborn as sns

RESULT_DIR = "tests/experiments/scale"

def plot_output1(*args):
    filenames = [os.path.basename(f) for f in args]
    jsons = []
    for f in args:
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

def plot_output3(file, depth):
    fig, ax = plt.subplots(1, 1)#, figsize=(4, 2.5))
    with open(file) as fh:
        oo = json.load(fh)
    oo = [o for o in oo if o["depth"] == depth]

    print(len(oo))

    x = [o["num_trees"] for o in oo]
    A = [util.get_best_astar(o["a*"]) for o in oo]
    ARA = [max(map(lambda b: b[1], o["ara*"]["solutions"])) for o in oo]
    merge = [min(map(lambda b: b[1], o["merge"]["bounds"])) for o in oo]

    relA = [1.0 for a in A]
    relARA = [ara/a for a, ara in zip(A, ARA)]
    relmerge = [m/a for a, m in zip(A, merge)]

    ax.fill_between(x, relA, relARA)
    ax.semilogx(x, relA, label="A*")
    ax.semilogx(x, relARA, label="ARA*")
    ax.semilogx(x, relmerge, label="merge")

    ax.set_xticks(x)
    ax.set_xticks(x, minor=True)
    ax.set_xticklabels([str(s) for s in x])

    ax.legend()
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
    ax.set_ylabel("states per sec.")

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

    l0, = ax.plot(num_vertices, [a-ara for a, ara in zip(A, ARA)], ".", alpha=0.05, zorder=-1, markersize=20)
    l1, = ax.plot(num_vertices, [m-a for a, m in zip(A, merge)], ".", alpha=0.05, zorder=-1, markersize=20)

    bins = np.linspace(min(num_vertices), max(num_vertices), 20)
    bin_width = bins[1]-bins[0]
    assignments = np.digitize(num_vertices, bins)

    meanA = [np.mean(merge[assignments==int(bin)]-A[assignments==int(bin)]) for bin in range(len(bins))]
    stdA = [np.std(merge[assignments==int(bin)]-A[assignments==int(bin)]) for bin in range(len(bins))]

    ax.bar(bins, meanA, 0.45*bin_width, yerr=stdA, color=l1.get_color())

    #for x, a, ara in zip(num_vertices, A, ARA):
    #    ax.plot([x, x], [a, ara], ".-b", alpha=0.25)
    #for x, a, m in zip(num_vertices, A, merge):
    #    ax.plot([x, x], [a, m], "-b", alpha=0.25)

    plt.show()

def time_to_beat_merge(o):
    A = [b[1]-b[0] for b in o["a*"]["bounds"]]
    At = o["a*"]["bounds_times"]
    merge = o["merge"]["bounds"][-1]
    merge = merge[1] - merge[0]

    print(A)
    print(At)
    print(merge)

    return [at for a, at in zip(A, At) if a < merge][0]

def plot_output6(pattern):
    oo = []
    for f in glob.glob(f"tests/experiments/scale/{pattern}"):
        with open(f) as fh:
            oo += json.load(fh)
    print(len(oo), "records")

    fig, ax = plt.subplots(1, 1)#, figsize=(4, 2.5))

    num_vertices = [o["a*"]["num_vertices0"] + o["a*"]["num_vertices1"] for o in oo]
    At = [util.get_best_astar(o["a*"]) for o in oo]
    mt = [o["merge"]["times"][-1] for o in oo]
    Ab = [time_to_beat_merge(o) for o in oo]

    ratio = np.array([m/a for a, m in zip(Ab, mt)])

    #l0, = ax.plot(num_vertices, ratio, ".")

    bins = np.linspace(min(num_vertices), max(num_vertices), 10)
    bin_width = bins[1]-bins[0]
    assignments = np.digitize(num_vertices, bins)

    meanA = [np.mean(ratio[assignments==int(bin)]) for bin in range(len(bins))]
    #stdA = [np.std(ratio[assignments==int(bin)]) for bin in range(len(bins))]

    #for b in range(5, len(bins)):
    #    data = ratio[assignments==int(b)]
    #    print(data)
    #    ax.boxplot(data)
    #    break

    ax.bar(bins, meanA, 0.45*bin_width)

    #sns.set(style="whitegrid", palette="pastel", color_codes=True)
    #data = pd.DataFrame({"bin": assignments, "value": ratio})
    #sns.violinplot(x="bin", y="value", data=data)
    plt.show()

if __name__ == "__main__":
    if int(sys.argv[1]) == 1:
        plot_output1(os.path.join("tests/experiments/scale", sys.argv[2]))
    if int(sys.argv[1]) == 2:
        plot_output2(os.path.join("tests/experiments/scale", sys.argv[2]), int(sys.argv[3]))
    if int(sys.argv[1]) == 3:
        plot_output3(os.path.join("tests/experiments/scale", sys.argv[2]), int(sys.argv[3]))
    if int(sys.argv[1]) == 4:
        plot_output4(os.path.join("tests/experiments/scale", sys.argv[2]), int(sys.argv[3]))
    if int(sys.argv[1]) == 5:
        plot_output5(sys.argv[2])
    if int(sys.argv[1]) == 6:
        plot_output6(sys.argv[2])
