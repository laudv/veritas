import sys, os, json, glob, gzip
import numpy as np
import pandas as pd
import util
from io import StringIO

import matplotlib.pyplot as plt


RESULT_DIR = "tests/experiments/scale"

def tikzhist(buf, plotname, data, bins):
    #bins = [-0.1, 1.0, 10, 100, 1000, 10000000]
    N = len(data)
    hist = np.histogram(data, bins=bins)
    print("histogram", plotname, hist[0], hist[1])
    print("histogram cumsum", plotname, np.cumsum(hist[0])/sum(hist[0]))
    c1, c2 = ("myblue", "black") if "gap" in plotname else ("black", "mygreen")

    def lf(l):
        return f"\\tiny {l}"

    #print( f"\\newcommand{{\\{plotname}}}[1]{{", file=buf)
    print(  "\\begin{tikzpicture}", file=buf)
    print("""\\begin{axis}[
    width=2.5cm,height=2.4cm,axis lines=left,
    ymin=-0.03,ymax=1.1,xmin=-1.0,xmax=5.5,
    xmajorgrids=false,ymajorgrids=true,yminorgrids=true,xmajorticks=false,""", file=buf)
    #print( f"    xtick={{{','.join(map(str, range(len(bins))))}}},", file=buf)
    #print( f"    xticklabels={{{','.join(map(lf, labels))}}},", file=buf)
    #print(  "    xticklabels={,,},", file=buf)
    if "gapm" in plotname or "rndttb" in plotname:
        print(  "    ytick={0,1},yticklabel style={xshift=10pt},ylabel near ticks,yticklabels={,,},minor y tick num=4,", file=buf)
    else:
        print(  "    ytick={0,1},yticklabel style={xshift=2pt},yticklabels={\\tiny 0,\\tiny 1},minor y tick num=4,", file=buf)
    print(  "    every axis plot/.append style={ybar,bar width=.6,bar shift=0,fill}", file=buf)
    print(  "]", file=buf)
    print(f"\\addplot[{c1}] coordinates {{({0}, {hist[0][0]/N:.2f})}};", file=buf)
    for i, v in enumerate(hist[0][1:]):
        print(f"\\addplot[{c2}] coordinates {{({i+1}, {v/N:.2f})}};", file=buf)
    print("\\end{axis}", file=buf)
    print("\\end{tikzpicture}", file=buf)
    #print("}", file=buf)

def tikzstandalone_open():
    buf = StringIO()
    print("\\documentclass[tikz=true]{standalone}", file=buf)
    print("\\usepackage{pgfplots}", file=buf)
    print("\\pgfplotsset{compat=1.16}", file=buf)
    print("\\pgfplotsset{grid style={dotted}}", file=buf)
    print("\\definecolor{myred}{RGB}{209,13,13}", file=buf)
    print("\\definecolor{myblue}{RGB}{54,89,194}", file=buf)
    print("\\definecolor{mygreen}{RGB}{49,128,0}", file=buf)

    print("\\begin{document}", file=buf)
    return buf

def tikzstandalone_close(buf):
    print("\\end{document}", file=buf)

def time_to_beat_merge(o):
    A = [b[1]-b[0] for b in o["a*"]["bounds"]]
    At = o["a*"]["bounds_times"]
    try:
        merge = o["merge"]["bounds"][-1][1][1] - o["merge"]["bounds"][-1][0][0]
    except:
        merge = o["merge"]["bounds"][-1][1] # - o["merge"]["bounds"][-1][0]

    #print(A)
    #print(At)
    #print(merge)

    try:
        return [at for a, at in zip(A, At) if a-merge < 1e-5][0]
    except:
        print("INF TTB for merge =", merge, ":", A)
        return np.inf

def random(divide=""):
    data = dict()

    for k, g in [
            ("A", "tests/experiments/scale/allstate/rnd4g30s10N_*"),
            ("CH", "tests/experiments/scale/calhouse/rnd4g30s10N_*"),
            ("CT", "tests/experiments/scale/covtype/rnd4g30s10N_*"),
            ("H", "tests/experiments/scale/higgs/rnd4g30s10N_*")]:
        oo = []
        for f in glob.glob(g):
            with open(f) as fh:
                oo += json.load(fh)
        print(len(oo), "records (", k, ")")
        if divide == "numtrees":
            for num_trees in sorted(set([o["num_trees"] for o in oo])):
                data[f"{k} {num_trees}"] = [o for o in oo
                        if o["num_trees"] == num_trees]
        elif divide == "depth":
            for depth in sorted(set([o["depth"] for o in oo])):
                data[f"{k} {depth}"] = [o for o in
                        oo if o["depth"] == depth]
        else:
            data[k] = oo

    cols = {
        "dataset": [],
        "num_experiments": [],
        "exact_ours": [],
        "exact_merge": [],
        "better_upper": [],
        "better_lower": [],
        "gap": [],
        "gapm": [],
        "ttb": [],
    }


    buf = tikzstandalone_open()
    figcount = 1

    for k, oo in data.items():
        if len(oo) == 0:
            print("skipping", k)
            continue
        print("%", k)
        A = [util.get_best_astar(o["a*"], task="both") for o in oo]
        ARA = [max(o["ara*"]["solutions"], key=lambda b: b[1]-b[0])
                if len(o["ara*"]["solutions"]) > 0
                else (np.inf, -np.inf)
                for o in oo]
        merge = [o["merge"]["bounds"][-1] for o in oo]

        num_experiments = len(oo)
        better_upper = [a[1] < m[1][1] or abs(a[1]-m[1][1])/a[1] < 1e-5 for a, m in zip(A, merge)]
        better_lower = [ara[1] > m[1][0] or abs(ara[1]-m[1][0])/ara[1] < 1e-5 for ara, m in zip(ARA, merge)]

        exact_ours = [len(o["a*"]["solutions"]) > 0
                or (len(o["ara*"]["epses"]) > 0 and o["ara*"]["epses"][-1] == 1.0)
                for o in oo]
        exact_merge = [o["merge"]["optimal"] for o in oo]

        #for o, a, ara, m in zip(oo, A, ARA, merge):
        #    if not (a[1] < m[1][1] or abs(a[1]-m[1][1])/a[1] < 1e-5):
        #        print("upper", a, m, abs(a[1] - m[1][1]), o["a*"]["num_vertices1"])
        #    if not (ara[1] > m[1][0] or abs(ara[1]-m[1][0])/ara[1] < 1e-5):
        #        print("lower", ara, m, abs(ara[1] - m[1][0]), o["ara*"]["num_vertices1"])

        #gap_ours = [2*abs((a[1]-ara[1])/(a[1]+ara[1])) for a, ara in zip(A, ARA)]
        #gap_ours = [ara[1]/a[1] for a, ara in zip(A, ARA)]
        gap_ours = [abs((a[1]-ara[1])/a[1]) for a, ara in zip(A, ARA)]
        gap_merge = [abs((m[1][1]-m[1][0])/m[1][1]) for m in merge]
        #for m, exact in zip(merge, exact_merge):
        #    if exact:
        #        print(m)


        ttb = [time_to_beat_merge(o) for o in oo]
        ttb_ratio = [o["merge"]["times"][-1]/(t0+1e-5) for t0,o in zip(ttb, oo)]

        #print(ttb)

        cols["dataset"].append(k)
        cols["num_experiments"].append(num_experiments)
        cols["exact_ours"].append(sum(exact_ours) / num_experiments * 100)
        cols["exact_merge"].append(sum(exact_merge) / num_experiments * 100)
        cols["better_upper"].append(sum(better_upper) / num_experiments * 100)
        cols["better_lower"].append(sum(better_lower) / num_experiments * 100)
        cols["gap"].append(f"\\myrndfig{divide}{{{figcount}}}")
        cols["gapm"].append(f"\\myrndfig{divide}{{{figcount+1}}}")
        cols["ttb"].append(f"\\myrndfig{divide}{{{figcount+2}}}")
        figcount += 3
        #cols["gap10"].append(sum(g < 0.1 for g in gap_ours) / num_experiments * 100)
        #cols["gap50"].append(sum(g < 0.5 for g in gap_ours) / num_experiments * 100)
        #cols["ttb1000"].append(sum(t >= 1000.0 for t in ttb) / num_experiments * 100)
        #cols["ttb100"].append(sum(t >= 100.0 for t in ttb) / num_experiments)
        #cols["ttb10"].append(sum(t >= 10.0 for t in ttb) / num_experiments * 100)

        gapbins = [-0.1, 0.01, 0.2, 0.5, 1.0, 10000000]

        tikzhist(buf, f"figrndgap{k}{divide}", gap_ours, bins=gapbins)
        tikzhist(buf, f"figrndgapm{k}{divide}", gap_merge, bins=gapbins)
        tikzhist(buf, f"figrndttb{k}{divide}", ttb_ratio, bins=[-0.1, 0.99999, 10, 100, 1000, 10000000])

    print("num_experiments", cols["num_experiments"])
    print("better_upper", cols["better_upper"])
    print("better_lower", cols["better_lower"])
    #print(cols)

    df = pd.DataFrame({k:v for k,v in cols.items() 
        if k in ["dataset", "exact_ours", "exact_merge", "gap", "gapm", "ttb"]})
    df.columns = ["Data", "Exact \\ouralg{}", "Exact \\merge{}", "Gap \\ouralg{}", "Gap \\merge{}", "TTB"]
    print(df)
    print(df.to_latex(index=False, float_format="%.1f\%%", escape=False))
    tikzstandalone_close(buf)
    if "IMG_OUTPUT" in os.environ:
        with open(os.path.join(os.environ["IMG_OUTPUT"], f"table_random_figures{divide}.tex"), "w") as f:
            print(buf.getvalue(), file=f)
        with open(os.path.join(os.environ["IMG_OUTPUT"], f"table_random{divide}.tex"), "w") as f:
            print(f"\\newcommand{{\\myrndfig{divide}}}[1]{{\\includegraphics[page=#1]{{images/table_random_figures{divide}.pdf}}}}", file=f)
            if divide:
                i = df.shape[0]//2
                print("i =", i)
                df.iloc[:i,:].to_latex(f, index=False, float_format="%.1f\%%", escape=False,
                        column_format="m{0.4cm}m{1.1cm}m{0.9cm}m{1.1cm}m{0.9cm}m{0.9cm}")
                print("\hspace{0.5cm}", file=f)
                df.iloc[i:,:].to_latex(f, index=False, float_format="%.1f\%%", escape=False,
                        column_format="m{0.4cm}m{1.1cm}m{0.9cm}m{1.1cm}m{0.9cm}m{0.9cm}")
            else:
                df.to_latex(f, index=False, float_format="%.1f\%%", escape=False,
                        column_format="m{0.4cm}m{1.1cm}m{0.9cm}m{1.1cm}m{0.9cm}m{0.9cm}")
        print(f"wrote table tex to {os.environ['IMG_OUTPUT']}")

def robust():
    cache_file = f"/tmp/temporary_mnist_robust_cache.h5"

    if not os.path.exists(cache_file):
        oo = []
        for f in glob.glob(f"tests/experiments/scale/mnist/robbb*"):
            print("loading", f)
            with gzip.open(f, "r") as fh:
                x = json.load(fh)
                print(len(x), end=" ")
                x = [o for o in x if len(o["merge"]["bounds"]) > 0]
                print(len(x))
                oo += x

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
        merge = [o["merge"]["bounds"][-1]
                if len(o["merge"]["bounds"]) > 0
                else (-np.inf, np.inf) for o in oo]
        data["merge0"] = [m[0] for m in merge]
        data["merge1"] = [m[1] for m in merge]
        data["A_time"] = [o["a*"]["bounds_times"][-1] for o in oo]
        data["ARA_time"] = [o["ara*"]["bounds_times"][-1] for o in oo]
        data["merge_time"] = [o["merge"]["times"][-1] for o in oo]
        data["ttb"] = [time_to_beat_merge(o) for o in oo]
        data["example_seed"] = [o["example_seed"] for o in oo]
        data["example_i"] = [o["example_i"] for o in oo]
        data["follow_a*"] = [o["follow_astar"] for o in oo]
        data["delta"] = [o["delta"] for o in oo]
        data["source"] = [o["example_label"] for o in oo]
        data["target"] = [o["target_label"] for o in oo]
        data["Aissol"] = [len(o["a*"]["solutions"]) > 0 for o in oo]
        data["mergeissol"] = [o["merge"]["optimal"] for o in oo]
        data["binsearch_step"] = [o["binsearch_step"] for o in oo]
        data["done_early"] = ["done_early" in o for o in oo]
        data["a_unsat"] = [a[0] > a[1] for a in A]
        data["merge_unsat"] = [m[0] > m[1] for m in merge]
        data["up"] = [o["binsearch_upper"] for o in oo]
        data["lo"] = [o["binsearch_lower"] for o in oo]


        df = pd.DataFrame(data)
        #df["a_unsat_merge"] = [a[0]> a[1] and m[0][0]<=m[1][1] for a, m in zip(A, merge)]
        #df["a_merge_unsat"] = [a[0]<=a[1] and m[0][0]> m[1][1] for a, m in zip(A, merge)]

        df.to_hdf(cache_file, key="data")
    else:
        df = pd.read_hdf(cache_file)

    #print(df)

    print(df.shape, df.columns)

    # how often is our output bound better than merge's?
    bound_merge = (df["merge1"]-df["merge0"])
    bound_A = (df["A1"]-df["A0"])
    better = (bound_merge > bound_A)
    same = (~better) & ((bound_A - bound_merge) < 1e-5)

    dfg = df.groupby(["example_seed", "example_i", "source", "follow_a*", "target"])

    agg = {
        "total": len(dfg)>>1,
        "win": 0,
        "games": df.shape[0],
        "same_output_bound": same.sum(),
        "better_output_bound": better.sum(),
        "same_delta_bound": 0,
        "better_delta_bound": 0,
        "aexact": 0,
        "mexact": 0,
    }
    ttb_ratio = []
    timediff = []
    slower_times = []

    def exact_solve_ours(g):
        all_unsats = g.loc[(g["a_unsat"])]["delta"]
        all_sats = g.loc[(g["Aissol"] | g["ARAeps"] == 1.0)]["delta"]
        if len(all_sats) > 0 and len(all_unsats) > 0:
            certainly_unsat = all_unsats.values[-1]
            certainly_sat = all_sats.values[-1]
            is_exact = np.ceil(certainly_unsat) == np.floor(certainly_sat)
            if not is_exact:
                print("exact_solve_ours", certainly_unsat, certainly_sat, is_exact)
            return is_exact
        return False

    def exact_solve_merge(g):
        all_unsats = g.loc[(g["merge_unsat"])]["delta"]
        all_sats = g.loc[(g["mergeissol"])]["delta"]
        if len(all_sats) > 0 and len(all_unsats) > 0:
            certainly_unsat = all_unsats.values[-1]
            certainly_sat = all_sats.values[-1]
            is_exact = np.ceil(certainly_unsat) == np.floor(certainly_sat)
            #print("exact_solve_ours", certainly_unsat, certainly_sat, is_exact)
            return is_exact
        return False

    #x = dfg["a_unsat_merge"].any() # was at any point in the search A* better than merge?
    for i0 in dfg.indices.keys():
        if not i0[3]: continue # only follow a*
        i1 = (i0[0], i0[1], i0[2], False, i0[4]) # follow merge

        #print(i0, i1)
        g0 = dfg.get_group(i0)
        g1 = dfg.get_group(i1)

        alo = g0["lo"].values[-1]
        mlo = g1["lo"].values[-1]
        same = np.ceil(alo) == np.ceil(mlo)
        better = np.ceil(alo) > np.ceil(mlo)
        agg["same_delta_bound"] += int(same)
        agg["better_delta_bound"] += int(better)
        #agg["win"] += len(g0.loc[(g0["a_unsat"] & ~g0["merge_unsat"])])
        #agg["win"] += len(g1.loc[(g0["a_unsat"] & ~g1["merge_unsat"])])
        ttb_ratio += [(t1+1e-5)/(t0+1e-5) for t0, t1 in zip(g0["ttb"], g0["merge_time"])]
        ttb_ratio += [(t1+1e-5)/(t0+1e-5) for t0, t1 in zip(g1["ttb"], g1["merge_time"])]
        slower_times += [t0 for t0, t1 in zip(g0["A_time"], g0["merge_time"]) if t1<t0 and abs(t0-t1) > 1e-5]
        slower_times += [t0 for t0, t1 in zip(g1["A_time"], g1["merge_time"]) if t1<t0 and abs(t0-t1) > 1e-5]
        #timediff +=  [t1-t0 for t0, t1 in zip(g0["A_time"], g0["merge_time"])]
        #timediff +=  [t1-t0 for t0, t1 in zip(g1["A_time"], g1["merge_time"])]

        agg["aexact"] += int(exact_solve_ours(g0))
        agg["mexact"] += int(exact_solve_merge(g1))

    agg["aexact"] /= agg["total"]
    agg["mexact"] /= agg["total"]
    agg["same_delta_bound"] /= agg["total"]
    agg["better_delta_bound"] /= agg["total"]
    agg["same_output_bound"] /= df.shape[0]
    agg["better_output_bound"] /= df.shape[0]
    agg["aexact"] *= 100
    agg["mexact"] *= 100
    agg["same_delta_bound"] *= 100
    agg["better_delta_bound"] *= 100
    agg["same_output_bound"] *= 100
    agg["better_output_bound"] *= 100
    print("agg", agg)

    bins = [-0.1, 1.0, 10, 100, 1000, 10000000]
    hist = np.histogram(ttb_ratio, bins=bins)
    print(hist[0], len(ttb_ratio))

    column_selection = ["better_output_bound", "better_delta_bound", "aexact", "mexact"]
    dfagg = pd.DataFrame({k: [agg[k]] for k in column_selection})

    buf = tikzstandalone_open()
    tikzhist(buf, f"figrobustttb", ttb_ratio, bins=[-0.1, 1.0, 10, 100, 1000, 10000000])
    tikzstandalone_close(buf)

    dfagg["hist"] = "\\myrobfig{1}"
    dfagg.columns = ["better $\\bar{b}$", "better $\\ubar{\\delta}$", "\\ouralg{} exact", "\\merge{} exact", "TTB"]
    print(dfagg)
    print(dfagg.to_latex(index=False, float_format="%.1f\%%", escape=False))

    if "IMG_OUTPUT" in os.environ:
        with open(os.path.join(os.environ["IMG_OUTPUT"], "table_robust_figures.tex"), "w") as f:
            print(buf.getvalue(), file=f)
        #with open(os.path.join(os.environ["IMG_OUTPUT"], "table_robust.tex"), "w") as f:
        #    print("\\newcommand{\\myrobfig}[1]{\\includegraphics[page=#1]{images/table_robust_figures.pdf}}", file=f)
        #    dfagg.to_latex(f, index=False, float_format="%.1f\%%", escape=False,
        #            column_format="m{1cm}m{1cm}m{1.2cm}m{1cm}m{1.5cm}")
        print(f"wrote table tex to {os.environ['IMG_OUTPUT']}")



if __name__ == "__main__":
    if sys.argv[1] == "random":
        random()
    if sys.argv[1] == "random_numtrees":
        random(divide="numtrees")
    if sys.argv[1] == "random_depth":
        random(divide="depth")
    if sys.argv[1] == "robust":
        robust()
