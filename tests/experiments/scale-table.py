import sys, os, json, glob, gzip
import numpy as np
import pandas as pd
import util
from io import StringIO

import matplotlib.pyplot as plt


RESULT_DIR = "tests/experiments/scale"

def time_to_beat_merge(o):
    A = [b[1] for b in o["a*"]["bounds"]]
    At = o["a*"]["bounds_times"]
    merge = o["merge"]["bounds"][-1][1][1]

    #print(A)
    #print(At)
    #print(merge)

    try:
        return [at for a, at in zip(A, At) if a < merge][0]
    except:
        return np.inf

def random():
    data = dict()

    for k, g in [
            ("allstate", "tests/experiments/scale/allstate/rnd4g30s10N_*"),
            ("calhouse", "tests/experiments/scale/calhouse/rnd4g30s10N_*"),
            ("covtype", "tests/experiments/scale/covtype/rnd4g30s10N_*"),
            ("higgs", "tests/experiments/scale/higgs/rnd4g30s10N_*")]:
        oo = []
        for f in glob.glob(g):
            with open(f) as fh:
                oo += json.load(fh)
        print(len(oo), "records (", k, ")")
        data[k] = oo

    cols = {
        "dataset": [],
        "num_experiments": [],
        "exact_ours": [],
        "exact_merge": [],
        "better_upper": [],
        "better_lower": [],
        "gap10": [],
        "gap50": [],
        "ttb1000": [],
        #"ttb100": [],
        "ttb10": [],
    }

    for k, oo in data.items():
        print(k)
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
        #gap_merge = [2*abs((m[1][1]-m[1][0])/(m[1][1]+m[1][0])) for m in merge]
        #for m, exact in zip(merge, exact_merge):
        #    if exact:
        #        print(m)


        ttb = [time_to_beat_merge(o) for o in oo]
        ttb = [o["merge"]["times"][-1] / ta if ta > 0.0 else 9999 for ta, o in zip(ttb, oo)]

        #print(ttb)

        cols["dataset"].append(k)
        cols["num_experiments"].append(num_experiments)
        cols["exact_ours"].append(sum(exact_ours) / num_experiments * 100)
        cols["exact_merge"].append(sum(exact_merge) / num_experiments * 100)
        cols["better_upper"].append(sum(better_upper) / num_experiments * 100)
        cols["better_lower"].append(sum(better_lower) / num_experiments * 100)
        cols["gap10"].append(sum(g < 0.1 for g in gap_ours) / num_experiments * 100)
        cols["gap50"].append(sum(g < 0.5 for g in gap_ours) / num_experiments * 100)
        cols["ttb1000"].append(sum(t >= 1000.0 for t in ttb) / num_experiments * 100)
        #cols["ttb100"].append(sum(t >= 100.0 for t in ttb) / num_experiments)
        cols["ttb10"].append(sum(t >= 10.0 for t in ttb) / num_experiments * 100)

    print(cols)

    df = pd.DataFrame(cols)
    #print(df)
    print(df.to_latex(index=False, float_format="%.1f"))

def robust():
    cache_file = f"/tmp/temporary_mnist_robust_cache.h5"

    if not os.path.exists(cache_file):
        oo = []
        for f in glob.glob(f"tests/experiments/scale/mnist/rob2g10s*"):
            with gzip.open(f, "r") as fh:
                oo += json.load(fh)

        print(len(oo))
        print(list(oo[0].keys()))
        print(list(oo[0]["a*"].keys()))

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
        data["merge0"] = [m[0][0] for m in merge]
        data["merge1"] = [m[1][1] for m in merge]
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
        data["mergeissol"] = [o["merge"]["bounds"][0]==o["merge"]["bounds"][1] for o in oo]
        data["binsearch_step"] = [o["binsearch_step"] for o in oo]
        data["done_early"] = ["done_early" in o for o in oo]
        data["a_unsat"] = [a[0] > a[1] for a in A]
        data["merge_unsat"] = [m[0][0] > m[1][1] for m in merge]
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

    dfg = df.groupby(["example_seed", "example_i", "source", "follow_a*", "target"])

    agg = {
        "total": len(dfg)>>1,
        "win": 0,
        "games": df.shape[0],
        "same": 0,
        "better": 0,
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
            #print("exact_solve_ours", certainly_unsat, certainly_sat, is_exact)
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
        agg["same"] += int(same)
        agg["better"] += int(better)
        #agg["win"] += len(g0.loc[(g0["a_unsat"] & ~g0["merge_unsat"])])
        #agg["win"] += len(g1.loc[(g0["a_unsat"] & ~g1["merge_unsat"])])
        ttb_ratio += [t1/(t0+1e-5) for t0, t1 in zip(g0["ttb"], g0["merge_time"])]
        ttb_ratio += [t1/(t0+1e-5) for t0, t1 in zip(g1["ttb"], g1["merge_time"])]
        slower_times += [t0 for t0, t1 in zip(g0["A_time"], g0["merge_time"]) if t1<t0 and abs(t0-t1) > 1e-5]
        slower_times += [t0 for t0, t1 in zip(g1["A_time"], g1["merge_time"]) if t1<t0 and abs(t0-t1) > 1e-5]
        #timediff +=  [t1-t0 for t0, t1 in zip(g0["A_time"], g0["merge_time"])]
        #timediff +=  [t1-t0 for t0, t1 in zip(g1["A_time"], g1["merge_time"])]

        agg["aexact"] += int(exact_solve_ours(g0))
        agg["mexact"] += int(exact_solve_merge(g1))

    agg["aexact"] /= agg["total"]
    agg["mexact"] /= agg["total"]
    agg["same"] /= agg["total"]
    agg["better"] /= agg["total"]
    agg["aexact"] *= 100
    agg["mexact"] *= 100
    agg["same"] *= 100
    agg["better"] *= 100
    print(agg)

    bins = [-0.1, 1.0, 10, 100, 1000, 10000000]
    hist = np.histogram(ttb_ratio, bins=bins)
    print(hist[0], len(ttb_ratio))

    #print("len slower_times", len(slower_times))
    #plt.hist(slower_times, bins=1000)
    #plt.show()

    dfagg = pd.DataFrame({k: [v] for k, v in agg.items()})
    dfagg = dfagg.drop(columns=["games", "win", "total"])
    #print(dfagg)

    f = StringIO()
    scale = 3.0
    print("\\newcommand{\\slrobust}[0]{", file=f)
    print("\\definecolor{sparkspikecolor}{named}{red}%", file=f)
    print("\\begin{sparkline}{0.5}", file=f)
    print(f"    \\sparkspike 0.0 {scale*hist[0][0]/len(ttb_ratio):.3f}", file=f)
    print("\\end{sparkline}%", file=f)
    print("\\definecolor{sparkspikecolor}{named}{black}%", file=f)
    print(f"\\begin{{sparkline}}{{{len(hist[0])}}}", file=f)
    for i, v in enumerate(hist[0][1:]):
        print(f"    \\sparkspike {i*0.167+0.083:.3f} {scale*v/len(ttb_ratio):.3f}", file=f)
    print("\\end{sparkline}", file=f)
    print("}", file=f)

    dfagg["hist"] = "\\slrobust{}"
    dfagg.columns = ["equal $\\ubar{\\delta}$", "better $\\ubar{\\delta}$", "ours exact", "merge exact", "speed"]
    print(f.getvalue())
    print(dfagg.to_latex(index=False, float_format="%.1f\%%", escape=False))


if __name__ == "__main__":
    if sys.argv[1] == "random":
        random()
    if sys.argv[1] == "robust":
        robust()