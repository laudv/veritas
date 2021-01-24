import sys
import json, gzip

import numpy as np

import datasets

def x(jsons):
    mnist = datasets.Mnist()
    mnist.load_dataset()
    for j in jsons:
        example_i = j["example_i"]
        example = list(mnist.X.iloc[example_i, :])

        ver_norm = j["veritas_deltas"][-1][1]
        mer_norm = j["merge_deltas"][-1][1]
        kan_norm = np.round(max(abs(x-y) for x, y in zip(example, j["kantchelian"]["example"])))
        if ver_norm != mer_norm:
            print(f"example_i {example_i}: {j['example_label']} vs {j['target_label']}")
            print("   TIME: ", sum(x["total_time"] for x in j["veritas_log"]),
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

    jsons = parse_robustness_files(filename)

    if task == "x":
        x(jsons)
