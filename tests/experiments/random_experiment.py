import os, sys, json, gzip
import util
import datasets
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas.kantchelian import KantchelianOutputOpt
import numpy as np

MAX_TIME = 20
MAX_MEM = 4*1024*1024*1024

def random_experiment(dataset, num_trees, tree_depth, outfile, n, constraints_seed, algos):
    rng = np.random.RandomState(constraints_seed)

    for prune_seed in rng.randint(0, 2**31, size=n):
        prune_seed = int(prune_seed)
        result = {
            "algos": algos,
            "num_trees": num_trees,
            "tree_depth": tree_depth,
            "n": n,
            "constraints_seed": constraints_seed,
            "prune_seed": prune_seed,
        }

        dataset.load_dataset()
        dataset.load_model(num_trees, tree_depth)
        at = dataset.at
        opt = Optimizer(maximize=at, max_memory=MAX_MEM)
        result["num_vertices_before"] = opt.g1.num_vertices()
        box, target_num_vertices = util.randomly_prune_opt(dataset.X, opt, prune_seed)
        result["num_vertices_after"] = opt.g1.num_vertices()
        result["target_num_vertices"] = target_num_vertices
        print("target_num_vertices", target_num_vertices, "-", result["num_vertices_after"])

        if algos[0] == "1":
            print("\n== VERITAS ======================================")
            opt = Optimizer(maximize=at, max_memory=MAX_MEM)
            opt.prune_box(box, instance=1)
            dur, oom = opt.astar(max_time=MAX_TIME)
            result["veritas"] = opt.stats()
            result["veritas_time"] = dur
            result["veritas_oom"] = oom
            print("   ", result["veritas"]["bounds"][-1], dur)
            if len(result["veritas"]["solutions"]) > 0:
                print("    sol:", result["veritas"]["solutions"][0])
            print("    veritas time", dur)

            print("\n== VERITAS ARA* =================================")
            opt = Optimizer(maximize=at, max_memory=MAX_MEM)
            opt.prune_box(box, instance=1)
            dur, oom = opt.arastar(max_time=MAX_TIME)
            result["veritas_ara"] = opt.stats()
            result["veritas_ara_time"] = dur
            result["veritas_ara_oom"] = oom
            print("   ", result["veritas_ara"]["bounds"][-1], dur)
            if len(result["veritas_ara"]["solutions"]) > 0:
                print("    sol:", max(result["veritas_ara"]["solutions"], key=lambda s: s[1])[1])
            print("    veritas time", dur)

        if algos[1] == "1":
            print("\n== MERGE ========================================")
            opt = Optimizer(maximize=at, max_memory=MAX_MEM)
            opt.prune_box(box, instance=1)
            data = opt.merge(max_time=MAX_TIME)
            result["merge"] = data
            print("    merge time", data["total_time"])

        if algos[2] == "1":
            print("\n== KANTCHELIAN MIPS =============================")
            kan = KantchelianOutputOpt(at, max_time=MAX_TIME)
            kan.constraint_to_box(box)
            kan.optimize()

            result["kantchelian_output"] = kan.solution()
            result["kantchelian"] = kan.stats()
            print("   ", result["kantchelian"]["bounds"][-1], kan.total_time_p)
            print("   ", result["kantchelian_output"])

        result_str = json.dumps(result)
        result_bytes = result_str.encode('utf-8')
        outfile.write(result_bytes)
        outfile.write(b"\n")

        # Make sure we write so we don't lose anything on error
        outfile.flush()
        os.fsync(outfile)

if __name__ == "__main__":
    dataset = sys.argv[1]
    num_trees = int(sys.argv[2])
    tree_depth = int(sys.argv[3])
    outfile_base = sys.argv[4]
    constraints_seed = int(sys.argv[5])
    n = int(sys.argv[6])
    algos = sys.argv[7]
    assert len(algos) == 3
    outfile = f"{outfile_base}-{dataset}-seed{constraints_seed}-n{n}-{algos}.gz"

    #if dataset == "mnist":
    #    dataset = datasets.Mnist()
    if dataset == "calhouse":
        dataset = datasets.Calhouse()
    elif dataset == "allstate":
        dataset = datasets.Allstate()
    elif dataset == "covtype":
        dataset = datasets.Covtype()
    elif dataset == "higgs":
        dataset = datasets.Higgs()
    else:
        raise RuntimeError("invalid dataset")

    if "--yes" not in sys.argv and os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        try:
            random_experiment(dataset, num_trees, tree_depth, f, n, constraints_seed, algos)
        finally:
            print("results written to", outfile)
