import os, sys, json, gzip
import datasets
from veritas0 import Optimizer
from veritas0 import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas0.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np

import xgboost as xgb
from veritas0.xgb import addtrees_from_multiclass_xgb_model

MAX_TIME = 4.0


def robustness_experiment(example_is, outfile, algos):
    mnist = datasets.Mnist()
    #mnist.load_model(num_trees, tree_depth)
    mnist.load_dataset()

    bst = xgb.Booster()
    bst.load_model("/tmp/natural_mnist_0200.model")
    ats = addtrees_from_multiclass_xgb_model(bst, 10, lambda x: x)

    for example_i in example_is:
        example = list(mnist.X.iloc[example_i,:])
        example_label = int(mnist.y[example_i])
        at0 = ats[example_label]
        #for target_label in [j for j in range(10) if j!=example_label]:
        for target_label in [4]:
            at1 = ats[target_label]

            print(f"\n\n== EXAMPLE {example_i}: {example_label} vs {target_label} ({algos}) ===========")
            result = {
                "example_i": example_i,
                "example_label": example_label,
                "target_label": target_label,
                "algos": algos,
            }

            if algos[0] != "0":
                print("\n== VERITAS ======================================")
                ver = VeritasRobustnessSearch(at0, at1, example, start_delta=20,
                        eps_start=1.0, eps_incr=0.1,
                        max_time=MAX_TIME,
                        stop_condition=RobustnessSearch.INT_STOP_COND)
                ver_norm, ver_lo, ver_hi = ver.search()
                result["veritas_deltas"] = ver.delta_log
                result["veritas_log"] = ver.log
                result["veritas_time"] = ver.total_time
                result["veritas_time_p"] = ver.total_time_p
                result["veritas_examples"] = ver.generated_examples
                print("veritas time", ver.total_time, ver.total_time_p)

                print("\n== VERITAS ARA* =================================")
                ver = VeritasRobustnessSearch(at0, at1, example, start_delta=20,
                        eps_start=0.1, eps_incr=0.1,
                        max_time=MAX_TIME,
                        stop_condition=RobustnessSearch.INT_STOP_COND)
                ver_norm, ver_lo, ver_hi = ver.search()
                result["veritas_ara_deltas"] = ver.delta_log
                result["veritas_ara_log"] = ver.log
                result["veritas_ara_time"] = ver.total_time
                result["veritas_ara_time_p"] = ver.total_time_p
                result["veritas_ara_examples"] = ver.generated_examples
                print("veritas ARA* time", ver.total_time, ver.total_time_p)

            if algos[1] == "1":
                print("\n== MERGE ========================================")
                mer = MergeRobustnessSearch(at0, at1, example, max_merge_depth=2,
                        max_time=MAX_TIME,
                        start_delta=20, stop_condition=RobustnessSearch.INT_STOP_COND)
                mer_norm, mer_lo, mer_hi = mer.search()
                result["merge_deltas"] = mer.delta_log
                result["merge_log"] = mer.log
                result["merge_time"] = mer.total_time
                result["merge_time_p"] = mer.total_time_p
                print("merge time", mer.total_time, mer.total_time)

            if algos[1] == "3":
                print("\n== MERGE ========================================")
                mer = MergeRobustnessSearch(at0, at1, example, max_merge_depth=3,
                        max_time=MAX_TIME,
                        start_delta=20, stop_condition=RobustnessSearch.INT_STOP_COND)
                mer_norm, mer_lo, mer_hi = mer.search()
                result["merge_deltas"] = mer.delta_log
                result["merge_log"] = mer.log
                result["merge_time"] = mer.total_time
                result["merge_time_p"] = mer.total_time_p
                print("merge time", mer.total_time, mer.total_time)

            if algos[2] == "1":
                print("\n== TREECK =======================================")
                tck = TreeckRobustnessSearch(at0, at1, example, start_delta=20,
                        max_time=MAX_TIME,
                        stop_condition=RobustnessSearch.INT_STOP_COND)
                tck_norm, tck_lo, tck_hi = tck.search()
                result["treeck_deltas"] = tck.delta_log
                result["treeck_log"] = tck.log
                result["treeck_time"] = tck.total_time
                result["treeck_time_p"] = tck.total_time_p
                result["treeck_examples"] = tck.generated_examples,
                print("treeck time", tck.total_time, tck.total_time_p)

            if algos[3] == "1":
                print("\n== KANTCHELIAN MIPS =============================")
                kan = KantchelianTargetedAttack(at0, at1, example=example)
                kan.optimize()
                kan_example, kan_pred0, kan_pred1, kan_norm = kan.solution()
                result["kantchelian"] = kan.stats()
                result["kantchelian_example"] = kan_example
                result["kantchelian_pred"] = (kan_pred0, kan_pred1)
                result["kantchelian_delta"] = kan_norm
                print("kantchelian time", kan.total_time, kan.total_time_p)

            result_str = json.dumps(result)
            result_bytes = result_str.encode('utf-8')  
            outfile.write(result_bytes)
            outfile.write(b"\n")

            # Make sure we write so we don't lose anything on error
            outfile.flush()
            os.fsync(outfile)

if __name__ == "__main__":
    # examples for which kan performed badly (mostly 8vs4)
    example_is = [2068, 2001, 2063, 2097, 2057, 2043, 2013, 2077, 2002, 2026]
    outfile_base = sys.argv[1]
    algos = sys.argv[2] # algo order: veritas merge treeck kantchelian
    assert len(algos) == 4
    outfile = f"{outfile_base}-chenmodel-{algos}.gz"

    if "--yes" not in sys.argv and os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        try:
            robustness_experiment(example_is, f, algos)
        finally: 
            print("results written to", outfile)
