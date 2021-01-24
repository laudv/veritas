import os, sys, json, gzip
import datasets
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np

VERITAS_MAX_TIME = 2


def robustness_experiment(num_trees, tree_depth, example_is, outfile):
    mnist = datasets.Mnist()
    mnist.load_model(num_trees, tree_depth)
    mnist.load_dataset()

    for example_i in example_is:
        example = list(mnist.X.iloc[example_i,:])
        example_label = int(mnist.y[example_i])
        at0 = mnist.at[example_label]
        for target_label in [j for j in range(10) if j!=example_label]:
            at1 = mnist.at[target_label]

            print(f"\n\n== EXAMPLE {example_i}: {example_label} vs {target_label} ======")

            # VERITAS
            print("\n== VERITAS ======================================")
            ver = VeritasRobustnessSearch(at0, at1, example, start_delta=20,
                    max_time=VERITAS_MAX_TIME,
                    stop_condition=RobustnessSearch.INT_STOP_COND)
            ver_norm, ver_lo, ver_hi = ver.search()

            # MERGE
            print("\n== MERGE ========================================")
            mer = MergeRobustnessSearch(at0, at1, example, max_merge_depth=2,
                    max_time=VERITAS_MAX_TIME,
                    start_delta=20, stop_condition=RobustnessSearch.INT_STOP_COND)
            mer_norm, mer_lo, mer_hi = mer.search()

            # TREECK
            print("\n== TREECK =======================================")
            tck = TreeckRobustnessSearch(at0, at1, example, start_delta=20,
                    stop_condition=RobustnessSearch.INT_STOP_COND)
            tck_norm, tck_lo, tck_hi = tck.search()

            # Kantchelian MIPS
            print("\n== KANTCHELIAN MIPS =============================")
            kan = KantchelianTargetedAttack(at0, at1, example=example)
            kan.optimize()
            kan_example, kan_prediction0, kan_prediction1, kan_norm = kan.solution()

            result = {
                "example_i": example_i,
                "example_label": example_label,
                "target_label": target_label,
                "veritas_deltas": ver.delta_log,
                "veritas_log": ver.log,
                "veritas_examples": ver.generated_examples,
                "merge_deltas": mer.delta_log,
                "merge_log": mer.log,
                "treeck_deltas": tck.delta_log,
                "treeck_log": tck.log,
                "treeck_examples": tck.generated_examples,
                "kantchelian": { "time": kan.total_time, "example": kan_example,
                    "out0": kan_prediction0, "kan1": kan_prediction1 }
            }
            result_str = json.dumps(result)
            result_bytes = result_str.encode('utf-8')  
            outfile.write(result_bytes)
            outfile.write(b"\n")

            # Make sure we write so we don't lose anything on error
            outfile.flush()
            os.fsync(outfile)

if __name__ == "__main__":
    num_trees = int(sys.argv[1])
    tree_depth = int(sys.argv[2])
    example_is = range(*(int(i) for i in sys.argv[3].split(":")))
    outfile = f"{sys.argv[4]}{num_trees}-depth{tree_depth}-{example_is.start}:{example_is.stop}.gz"

    if os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        robustness_experiment(num_trees, tree_depth, example_is, f)

    print("results written to", outfile)
