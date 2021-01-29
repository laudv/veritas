import sys
import json, gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datasets

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
    mnist = datasets.Mnist()
    mnist.load_dataset()
    colnames = ["example_i", "label", "target_label", "ver_delta", "ver2_delta", "mer_delta",
            "tck_delta", "kan_delta", "ver_time", "ver2_time", "mer_time", "tck_time", "kan_time"]
    example_is = []
    example_labels = []
    target_labels = []
    ver_deltas, ver2_deltas, mer_deltas, tck_deltas, kan_deltas = [], [], [], [], []
    ver_times, ver2_times, mer_times, tck_times, kan_times = [], [], [], [], []
    for j in jsons:
        example_is.append(j["example_i"])
        example_labels.append(j["example_label"])
        target_labels.append(j["target_label"])
        if "veritas_deltas" in j:
            ver_deltas.append(np.ceil(j["veritas_deltas"][-1][1]))
            ver_times.append(j["veritas_time"])
        if "veritas_ara_deltas" in j:
            ver2_deltas.append(np.ceil(j["veritas_ara_deltas"][-1][1]))
            ver2_times.append(j["veritas_ara_time"])
        if "merge_deltas" in j:
            mer_deltas.append(np.ceil(j["merge_deltas"][-1][1]))
            mer_times.append(j["merge_time"])
        if "treeck_deltas" in j:
            tck_deltas.append(np.ceil(j["treeck_deltas"][-1][1]))
            tck_times.append(j["treeck_time"])
        if "kantchelian" in j:
            #print(list(j["kantchelian"].keys()))
            kan_deltas.append(np.round(j["kantchelian_delta"]))
            kan_times.append(j["kantchelian"]["time_p"])

        try:
            if ver_deltas[-1] > kan_deltas[-1]:
                print("???", j["veritas_deltas"][-1], kan_deltas[-1])
        except: pass

    columns = { k: v for k, v in zip(colnames, [example_is, example_labels,
        target_labels, ver_deltas, ver2_deltas, mer_deltas, tck_deltas, kan_deltas,
        ver_times, ver2_times, mer_times, tck_times, kan_times]) if len(v) > 0 }

    return pd.DataFrame(columns)
            
def plot(jsons):

    for j in jsons:
        #print(list(j.keys()))
        fig, ax = plt.subplots()

        ver_y = [b[0] for b in j["veritas_deltas"]]
        ver2_y = [b[0] for b in j["veritas_ara_deltas"]]
        try:
            ver_times = [b[3] for b in j["veritas_deltas"]]
            ver2_times = [b[3] for b in j["veritas_ara_deltas"]]
        except:
            print("approximating times for veritas")
            ver_times = np.linspace(0, j["veritas_time"], len(ver_y))
            ver2_times = np.linspace(0, j["veritas_ara_time"], len(ver2_y))


        #for l in j["veritas_log"]:
        #    print("A*", l["bounds"][-1], l["solutions"])
        #for l in j["veritas_ara_log"]:
        #    print("ARA*", l["bounds"][-1], l["solutions"])
        ver_total_opt_time = sum(b["total_time"] for b in j["veritas_log"])
        l, = ax.plot(ver_times, ver_y, label="veritas")
        ax.plot(ver2_times, ver2_y, label="veritas2", c=l.get_color(), ls="--")
        #print("veritas time actually optimizing:", j["veritas_time"], "vs", ver_total_opt_time,
        #        (j["veritas_time"]-ver_total_opt_time)/ver_total_opt_time)
        if "kantchelian" in j:
            kan_lo_y = [b[0] for b in j["kantchelian"]["bounds"]]
            kan_hi_y = [b[1] for b in j["kantchelian"]["bounds"]] + [j["kantchelian_delta"]]
            kan_times_lo = [t[1] for t in j["kantchelian"]["times"]]
            kan_times_hi = kan_times_lo + [j["kantchelian"]["time_p"]]
            ax.plot(kan_times_hi, kan_hi_y, label="milp")
            ax.plot(kan_times_lo, kan_lo_y, label="milp lo")
            ax.axhline(y=j["kantchelian_delta"], color="gray", ls=":")

        mer_y = [b[0] for b in j["merge_deltas"]]
        mer_times = np.linspace(0, j["merge_time"], len(mer_y))
        ax.plot(mer_times, mer_y, label="merge")

        ax.set_ylim([min(ver_y)-1, max(ver_y)+1])
        ax.set_xlabel("time")
        ax.set_ylabel("robustness \delta")
        ax.legend()
        plt.show()


def parse_file(filename):
    with gzip.open(filename, "rb") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]

def avg_time_90percentile(x):
    x = sorted(x)
    x = x[0:int(len(x)*0.9)]
    return np.mean(x)

if __name__ == "__main__":
    task = sys.argv[1]
    filenames = [f for f in sys.argv[2:] if not f.startswith("--")]
    jsons = [parse_file(f) for f in filenames]
    jsons = combine_results(*jsons)
    df = get_df(jsons)
    #print("how often ver better than mer", sum(df["ver_delta"]>df["mer_delta"]),df.shape[0])
    #print("how often mer better than ver", sum(df["ver_delta"]<df["mer_delta"]),df.shape[0])
    #print("how often ver exact", sum(df["ver_delta"]==df["kan_delta"]),df.shape[0])
    #print("how often mer exact", sum(df["mer_delta"]==df["kan_delta"]),df.shape[0])
    #print("mean times", df["ver_time"].mean(), df["mer_time"].mean(), df["kan_time"].mean())
    #print("mean times 90%", avg_time_90percentile(df["ver_time"]),
    #        avg_time_90percentile(df["mer_time"]),
    #        avg_time_90percentile(df["kan_time"]))
    print("mean delta diff ver", (df["kan_delta"]-df["ver_delta"]).abs().mean())
    print("mean delta diff ver2", (df["kan_delta"]-df["ver2_delta"]).abs().mean())
    print("mean delta diff mer", (df["kan_delta"]-df["mer_delta"]).abs().mean())
    print("ver closer to kan", sum((df["kan_delta"]-df["ver_delta"]).abs() <
            (df["kan_delta"]-df["mer_delta"]).abs()))
    print("ver2 closer to kan", sum((df["kan_delta"]-df["ver2_delta"]).abs() <
            (df["kan_delta"]-df["mer_delta"]).abs()))
    print("ver farther to kan", sum((df["kan_delta"]-df["ver_delta"]).abs() >
            (df["kan_delta"]-df["mer_delta"]).abs()))
    print("ver2 farther to kan", sum((df["kan_delta"]-df["ver2_delta"]).abs() >
            (df["kan_delta"]-df["mer_delta"]).abs()))
    print("mean time ver", df["ver_time"].mean())
    print("mean time ver2", df["ver2_time"].mean())
    print("mean time mer", df["mer_time"].mean())

    #plot(jsons)
    #plot([jsons[i] for i in ver_worse.index])
