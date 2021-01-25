import os, sys, json, gzip
import datasets
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack, KantchelianOutputOpt
import numpy as np

MAX_TIME = 120
MAX_MEM = 4*1024*1024*1024

def stress_experiment(dataset, outfile, algos):

    for num_trees, tree_depth in [
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 8),
            (100, 4),
            (100, 5),
            (100, 6),
            (100, 8),
            (200, 4),
            (200, 5),
            (200, 6),
            (200, 8),
            (300, 4),
            (300, 5),
            (300, 6),
            (300, 8),
            (400, 4),
            (400, 5),
            (400, 6),
            (400, 8),
            ]:
        dataset.load_model(num_trees, tree_depth)
        at = dataset.at
        if not isinstance(at, list):
            at = [at]

        for at_index, at in enumerate(at):
            result = {
                "at_index": at_index,
                "algos": algos,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
            }

            if algos[0] == "1":
                print("\n== VERITAS ======================================")
                opt = Optimizer(maximize=at, max_memory=MAX_MEM)
                dur, oom = opt.astar(max_time=MAX_TIME)
                result["veritas"] = opt.stats()
                result["veritas_time"] = dur
                result["veritas_oom"] = oom
                print("   ", result["veritas"]["bounds"][-1], dur)

            if algos[1] == "1":
                print("\n== MERGE ========================================")
                opt = Optimizer(maximize=at, max_memory=MAX_MEM)
                data = opt.merge(max_time=MAX_TIME)
                result["merge"] = data

            if algos[2] == "1":
                print("\n== KANTCHELIAN MIPS =============================")
                kan = KantchelianOutputOpt(at, max_time=MAX_TIME)
                kan.optimize()
                result["kantchelian"] = kan.solution()
                print("   ", result["kantchelian"], kan.total_time)

                print(kan.stats())

            result_str = json.dumps(result)
            result_bytes = result_str.encode('utf-8')
            outfile.write(result_bytes)
            outfile.write(b"\n")

            # Make sure we write so we don't lose anything on error
            outfile.flush()
            os.fsync(outfile)


if __name__ == "__main__":
    dataset = sys.argv[1]
    outfile_base = sys.argv[2]
    try: algos = sys.argv[3] # algo order: veritas merge treeck kantchelian
    except: algos="111"
    assert len(algos) == 3
    outfile = f"{outfile_base}-{dataset}-{algos}.gz"

    if dataset == "mnist":
        dataset = datasets.Mnist()
    if dataset == "calhouse":
        dataset = datasets.Calhouse()
    else:
        raise RuntimeError("invalid dataset")

    if os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        stress_experiment(dataset, f, algos)

    print("results written to", outfile)



