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

def x(jsons):
    mnist = datasets.Mnist()
    mnist.load_dataset()
    for j in jsons:
        example_i = j["example_i"]
        example = list(mnist.X.iloc[example_i, :])
        print(j["veritas_deltas"])
        ver_norm = np.ceil(j["veritas_deltas"][-1][0])
        mer_norm = j["merge_deltas"][-1][1]
        kan_norm = np.round(max(abs(x-y) for x, y in zip(example, j["kantchelian"]["example"])))
        #if ver_norm != mer_norm:
        print(f"example_i {example_i}: {j['example_label']} vs {j['target_label']}")
        print("   TIME: ", sum(x["total_time"] for x in j["veritas_log"]),
            j["veritas_time"],
            sum(x["total_time"] for x in j["merge_log"]),
            j["kantchelian"]["time"])
        print("   NORM:", ver_norm, mer_norm, kan_norm)

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
    x(jsons)
