import sys
import json, gzip

import numpy as np
import pandas as pd

import datasets

def zip_result(jsons1101, jsons2010):
    for j0, j1 in zip(jsons1101, jsons2010):
        for k, v in j1.items():
            if k.startswith("veritas"):
                j0[f"{k}_ara"] = v
            elif k not in j0:
                j0[k] = v
            elif k == "algos": pass
            else:
                assert j0[k] == v
        #print("1101", list(j0.keys()))
        #print("2010", list(j1.keys()))
    return jsons1101

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
            ver_times.append(j["veritas_time_p"])
        if "veritas_deltas_ara" in j:
            ver2_deltas.append(np.ceil(j["veritas_deltas_ara"][-1][1]))
            ver2_times.append(j["veritas_time_p_ara"])
        if "merge_deltas" in j:
            mer_deltas.append(np.ceil(j["merge_deltas"][-1][1]))
            mer_times.append(j["merge_time_p"])
        if "treeck_deltas" in j:
            tck_deltas.append(np.ceil(j["treeck_deltas"][-1][1]))
            tck_times.append(j["treeck_time_p"])
        if "kantchelian" in j:
            kan_deltas.append(np.round(j["kantchelian"]["norm"]))
            kan_times.append(j["kantchelian"]["time_p"])

        try:
            if ver_deltas[-1] > kan_deltas[-1]:
                print("???", j["veritas_deltas"][-1], kan_deltas[-1])
        except: pass

    columns = { k: v for k, v in zip(colnames, [example_is, example_labels,
        target_labels, ver_deltas, ver2_deltas, mer_deltas, tck_deltas, kan_deltas,
        ver_times, ver2_times, mer_times, tck_times, kan_times]) if len(v) > 0 }

    return pd.DataFrame(columns)
            


        #example_i = j["example_i"]
        #example = list(mnist.X.iloc[example_i, :])
        #print(j["veritas_deltas"])
        #ver_norm = np.ceil(j["veritas_deltas"][-1][0])
        #mer_norm = j["merge_deltas"][-1][1]
        #kan_norm = np.round(max(abs(x-y) for x, y in zip(example, j["kantchelian"]["example"])))
        ##if ver_norm != mer_norm:
        #print(f"example_i {example_i}: {j['example_label']} vs {j['target_label']}")
        #print("   TIME: ", sum(x["total_time"] for x in j["veritas_log"]),
        #    j["veritas_time"],
        #    sum(x["total_time"] for x in j["merge_log"]),
        #    j["kantchelian"]["time"])
        #print("   NORM:", ver_norm, mer_norm, kan_norm)

def parse_robustness_files(filename):
    with gzip.open(filename, "rb") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]

if __name__ == "__main__":

    task = sys.argv[1]
    filename = sys.argv[2]
    try: filename2 = sys.argv[3]
    except: filename2 = None

    print(filename, filename2)

    jsons = parse_robustness_files(filename)

    if filename2 is not None:
        jsons2 = parse_robustness_files(filename2)
        jsons = zip_result(jsons, jsons2)
    df = get_df(jsons)
    #print(df)
    #print(df[df["ver_delta"]>df["mer_delta"]])
    #print(df[df["kan_delta"]==df["mer_delta"]])
    #print(df[df["kan_delta"]==df["ver_delta"]])
    #print(df[df["ver_delta"] != df["ver2_delta"]])
    print("how often ver better than mer", sum(df["ver_delta"]>df["mer_delta"]),df.shape[0])
    print("how often ver exact", sum(df["ver_delta"]==df["kan_delta"]),df.shape[0])
    print("how often mer exact", sum(df["mer_delta"]==df["kan_delta"]),df.shape[0])
    print("mean times", df["ver_time"].mean(), df["mer_time"].mean(), df["kan_time"].mean())
    print("mean delta diff ver", (df["kan_delta"]-df["ver_delta"]).mean())
    print("mean delta diff mer", (df["kan_delta"]-df["mer_delta"]).mean())
