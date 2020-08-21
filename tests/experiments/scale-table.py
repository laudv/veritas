import sys, os, json, glob, gzip
import numpy as np
import pandas as pd
import util


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

        print(ttb)

        cols["dataset"].append(k)
        cols["num_experiments"].append(num_experiments)
        cols["exact_ours"].append(sum(exact_ours) / num_experiments)
        cols["exact_merge"].append(sum(exact_merge) / num_experiments)
        cols["better_upper"].append(sum(better_upper) / num_experiments)
        cols["better_lower"].append(sum(better_lower) / num_experiments)
        cols["gap10"].append(sum(g < 0.1 for g in gap_ours) / num_experiments)
        cols["gap50"].append(sum(g < 0.5 for g in gap_ours) / num_experiments)
        cols["ttb1000"].append(sum(t >= 1000.0 for t in ttb) / num_experiments)
        #cols["ttb100"].append(sum(t >= 100.0 for t in ttb) / num_experiments)
        cols["ttb10"].append(sum(t >= 10.0 for t in ttb) / num_experiments)

    print(cols)

    df = pd.DataFrame(cols)
    df.round(2)
    print(df)

random()
