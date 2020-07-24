import os, json
import numpy as np
import matplotlib.pyplot as plt
import util

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

if __name__ == "__main__":
    #plot_output2("tests/experiments/scale/soccer/testabc", 0)
    plot_output2("tests/experiments/scale/higgs/test100", 0)

