import os, sys, json, gzip
import subprocess
import datasets
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
from external_merge import external_merge
import numpy as np

def robustness_experiment(num_trees, tree_depth, example_is, max_time, outfile, algos):
    mnist = datasets.Mnist()
    mnist.load_model(num_trees, tree_depth)
    mnist.load_dataset()

    for example_i in example_is:
        example = list(mnist.X.iloc[example_i,:])
        example_label = int(mnist.y[example_i])
        at0 = mnist.at[example_label]
        for target_label in [j for j in range(10) if j!=example_label]:
            at1 = mnist.at[target_label]

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
                        max_time=max_time,
                        stop_condition=RobustnessSearch.NO_STOP_COND)
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
                        max_time=max_time,
                        stop_condition=RobustnessSearch.NO_STOP_COND)
                ver_norm, ver_lo, ver_hi = ver.search()
                result["veritas_ara_deltas"] = ver.delta_log
                result["veritas_ara_log"] = ver.log
                result["veritas_ara_time"] = ver.total_time
                result["veritas_ara_time_p"] = ver.total_time_p
                result["veritas_ara_examples"] = ver.generated_examples
                print("veritas ARA* time", ver.total_time, ver.total_time_p)

            if algos[1] == "1":
                print("\n== MERGE ========================================")
                mer = MergeRobustnessSearch(at0, at1, example, max_merge_depth=999,
                        max_time=max_time,
                        start_delta=20, stop_condition=RobustnessSearch.NO_STOP_COND)
                mer_norm, mer_lo, mer_hi = mer.search()
                result["merge_deltas"] = mer.delta_log
                result["merge_log"] = mer.log
                result["merge_time"] = mer.total_time
                result["merge_time_p"] = mer.total_time_p
                print("merge time", mer.total_time, mer.total_time)

            if algos[1] == "e": # external
                print("\n== MERGE (external) =============================")
                deltas, times, exc = external_merge(mnist.model,
                        mnist.meta["columns"], example, example_label,
                        target_label, start_delta=40, max_level=2,
                        num_classes=10)
                result["merge_ext"] = {
                        "deltas": deltas,
                        "times": times,
                        "exc": exc
                }

            #if algos[1] == "3":
            #    print("\n== MERGE ========================================")
            #    mer = MergeRobustnessSearch(at0, at1, example, max_merge_depth=3,
            #            max_time=max_time,
            #            start_delta=20, stop_condition=RobustnessSearch.NO_STOP_COND)
            #    mer_norm, mer_lo, mer_hi = mer.search()
            #    result["merge_deltas"] = mer.delta_log
            #    result["merge_log"] = mer.log
            #    result["merge_time"] = mer.total_time
            #    result["merge_time_p"] = mer.total_time_p
            #    print("merge time", mer.total_time, mer.total_time)

            if algos[2] == "1":
                print("\n== TREECK =======================================")
                tck = TreeckRobustnessSearch(at0, at1, example, start_delta=20,
                        max_time=max_time,
                        stop_condition=RobustnessSearch.NO_STOP_COND)
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
    num_trees = int(sys.argv[1])
    tree_depth = int(sys.argv[2])
    example_is = range(*(int(i) for i in sys.argv[3].split(":")))
    outfile_base = sys.argv[4]
    max_time = int(sys.argv[5])
    algos = sys.argv[6] # algo order: veritas merge treeck kantchelian
    assert len(algos) == 4
    outfile = f"{outfile_base}{num_trees}-depth{tree_depth}-time{max_time}-{example_is.start}:{example_is.stop}-{algos}.gz"

    if "--yes" not in sys.argv and os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        try:
            robustness_experiment(num_trees, tree_depth, example_is, max_time,
                f, algos)
        finally: 
            print("results written to", outfile)
