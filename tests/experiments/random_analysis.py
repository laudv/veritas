import sys, gzip, json
import pandas as pd

import matplotlib.pyplot as plt

def combine_results(*jsons):
    jsons0 = []
    for js in zip(*jsons):
        j0 = {}
        for j in js:
            for k, v in j.items():
                j0[k] = v
        jsons0.append(j0)
    return jsons0

def get_df(jsons):

    col_names = ["num_trees", "tree_depth", "num_vert", "ver_bnd",
            "mer_bnd", "kan_bnd", "ver_time", "mer_time", "kan_time",
            "mer_oom"]
    num_trees, tree_depth, num_vertices = [], [], []
    ver_bnd, ver_time = [], []
    mer_bnd, mer_time, mer_oom = [], [], []
    kan_bnd, kan_time = [], []

    for j in jsons:
        #print(list(j.keys()))
        num_trees.append(j["num_trees"])
        tree_depth.append(j["tree_depth"])
        if "num_vertices_after" in j:
            num_vertices.append(j["num_vertices_after"])

        if "veritas" in j:
            #print(list(j["veritas"].keys()))
            if len(j["veritas"]["solutions"]) > 0:
                bnd = j["veritas"]["solutions"][0]
            else:
                bnd = j["veritas"]["bounds"][-1]
            ver_bnd.append(bnd)
            ver_time.append(j["veritas"]["times"][-1])

        if "merge" in j:
            #print(list(j["merge"].keys()))
            mer_bnd.append(j["merge"]["bounds"][-1][1])
            mer_time.append(j["merge"]["total_time"])
            mer_oom.append(j["merge"]["oom"])

        if "kantchelian" in j:
            #print(list(j["kantchelian"].keys()))
            #kan_bnd.append(min(j["kantchelian"]["bounds"][-1][1], j["kantchelian_output"]))
            kan_bnd.append(j["kantchelian"]["bounds"][-1][1])
            kan_time.append(j["kantchelian"]["time_p"])

    columns = { k: v for k, v in zip(col_names, [num_trees, tree_depth,
        num_vertices, ver_bnd, mer_bnd, kan_bnd, ver_time, mer_time, kan_time,
        mer_oom]) if len(v) > 0 }
    df = pd.DataFrame(columns)

    df["ver_faster"] = (df["kan_time"] > df["ver_time"]) & (df["ver_time"] < 20.0)
    df["ver_better"] = df["kan_bnd"].round(4) > df["ver_bnd"].round(4)
    df["ver_worse"] = df["kan_bnd"].round(4) < df["ver_bnd"].round(4)
    df["ver_equal"] = df["kan_bnd"].round(4) == df["ver_bnd"].round(4)

    return df.round(1)

def plot(jsons):

    for j in jsons:
        fig, ax = plt.subplots()
        ax.set_title(f"num_trees {j['num_trees']}, depth {j['tree_depth']}")

        if "veritas" in j:
            ver_lo_bnds = [b[0] for b in j["veritas"]["bounds"]]
            ver_up1_bnds = [b[1] for b in j["veritas"]["bounds"]]
            ver_up2_bnds = [b[2] for b in j["veritas"]["bounds"]]
            lv, = ax.plot(j["veritas"]["times"], ver_up1_bnds, label="veritas")
            ax.plot(j["veritas"]["times"], ver_lo_bnds, label="veritas lo", c=lv.get_color(), ls="--", marker="v")
            ax.plot(j["veritas"]["times"], ver_up2_bnds, label="veritas up2", c=lv.get_color(), ls=":", marker="^")

            print("== veritas bounds", j["veritas"]["bounds"][-1])
            print("heap_size", j["veritas"]["heap_size"])
            print("number of solutions veritas", len(j["veritas"]["solutions"]))

            if len(j["veritas"]["solutions"]) > 0:
                print("veritas solution", j["veritas"]["solutions"][0])
                #ax.axhline(y=j["veritas"]["solutions"][0]["output"], color="gray", ls=":", label="ver sol")

                #sols = sorted(j["veritas"]["solutions"], key=lambda p: p["time"])
                #ax.plot([s["time"] for s in sols], [s["output"] for s in sols], ":x",
                #        color=lv.get_color(), label="veritas ARA*")


        if "kantchelian" in j:
            kan_lo_bnds = [b[0] for b in j["kantchelian"]["bounds"]]
            #kan_hi_bnds = [b[1] for b in j["kantchelian"]["bounds"]] + [j["kantchelian_output"]]
            kan_hi_bnds = [b[1] for b in j["kantchelian"]["bounds"]]
            kan_times_lo = [t[1] for t in j["kantchelian"]["times"]]
            kan_times_hi = (kan_times_lo + [j["kantchelian"]["time_p"]])[0:len(kan_hi_bnds)]
            lk, = ax.plot(kan_times_hi, kan_hi_bnds, label="milp")
            ax.plot(kan_times_lo, kan_lo_bnds, label="milp lo", c=lk.get_color(), ls="--")
            if kan_lo_bnds[-1] == kan_hi_bnds[-1]:
                ax.axhline(j["kantchelian_output"], ls="--", color="lightgray", label="solution kan")
            print("== kantchelian_bounds", j["kantchelian"]["bounds"][-1], "time", j["kantchelian"]["time_p"])

        if "merge" in j:
            ax.plot(j["merge"]["times"], [b[1] for b in j["merge"]["bounds"]], label="merge")

        if "veritas0" in j:
            ver0_bnds = [b[1] for b in j["veritas0"]["bounds"]]
            lv0, = ax.plot(j["veritas0"]["times"], ver0_bnds, label="veritas0")

            print("veritas0", j["veritas0"]["bounds"][-1])
            if "veritas0_ara" in j and len(j["veritas0_ara"]["solutions"]) > 0:
                print("veritas0 sol", j["veritas0_ara"]["solutions"][-1])
                ax.plot(j["veritas0_ara"]["sol_times"], [s[1] for s in
                    j["veritas0_ara"]["solutions"]], label="veritas0 lo",
                    color=lv0.get_color(), ls="--", marker="x")
                #m = min(s for s in j["veritas0_ara"]["solutions"])
                #M = max(s for s in j["veritas0_ara"]["solutions"])
                #ax.plot([j["veritas0_ara"]["sol_times"][-1],
                #    j["veritas0"]["times"][-1]], [M, M], color=lv.get_color(), ls="--")

        if "kantchelian" in j and len(kan_lo_bnds) > 1:
            ax.set_ylim([kan_lo_bnds[1], max(ver_up1_bnds)])
        ax.set_xlabel("time")
        ax.set_ylabel("model output")
        ax.legend()
        plt.show()

        print()

def parse_file(filename):
    with gzip.open(filename, "rb") as f:
        lines = f.readlines()
        print(filename, len(lines))
        return [json.loads(line) for line in lines]

if __name__ == "__main__":
    filenames = [f for f in sys.argv[1:] if not f.startswith("--")]
    jsons = [parse_file(f) for f in filenames]
    #print(jsons)
    jsons = combine_results(*jsons)
    #print(jsons)
    #df = get_df(jsons)
    #print(df)
    #print(df.head(20))
    #print(f"Veritas faster than MILP:  {100.0*sum(df['ver_faster'])/len(df):4.0f}%   n={len(df)}")
    #print(f"Veritas better than MILP:  {100.0*sum(df['ver_better'])/len(df):4.0f}%")
    #print(f"Veritas worse than MILP:   {100.0*sum(df['ver_worse'])/len(df):4.0f}%")
    #print(f"Veritas as good as MILP:   {100.0*sum(df['ver_equal'])/len(df):4.0f}%")
    #print(f"Veritas better and faster: {100.0*sum(df['ver_better'] & df['ver_faster'])/len(df):4.0f}%")
    plot(jsons)
