import os, sys, json, gzip
#import util
import datasets
import veritas
from veritas import NodeSearch, Domain
#import veritas0
#from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
#from treeck_robust import TreeckRobustnessSearch
#from veritas.kantchelian import KantchelianOutputOpt
import numpy as np

MAX_TIME = 10
MAX_MEM = 4*1024*1024*1024

def _veritas_at_to_veritas0_at(veritas_at):
    at0 = veritas_at
    at1 = veritas0.AddTree()

    at1.base_score = at1.base_score

    for tree_index in range(len(at0)):
        tree0 = at0[tree_index]
        tree1 = at1.add_tree()
        stack = [(tree0.root(), tree1.root())]
        while len(stack) > 0:
            node0, node1 = stack.pop()
            if tree0.is_internal(node0):
                split = tree0.get_split(node0)
                if isinstance(split, veritas.LtSplit):
                    tree1.split(node1, split.feat_id, split.split_value)
                else:
                    tree1.split(node1, split.feat_id)
                stack.append((tree0.right(node0), tree1.right(node1)))
                stack.append((tree0.left(node0), tree1.left(node1)))
            else:
                tree1.set_leaf_value(node1, tree0.get_leaf_value(node0))

    return at1

def generate_random_constraints(X, num_constraints, seed):
    K = X.shape[1]
    m = X.min(axis=0)
    M = X.max(axis=0)

    rng = np.random.RandomState(seed)

    constraints = [Domain(m, M) for m, M in zip(m, M)]

    maybe_binary = [m == 0.0 and M == 1.0 for m, M in zip(m, M)]
    binary = [False for _ in range(K)]
    #print(binary)
    #print(X.shape)
    #for k in range(K):
    #    if not maybe_binary[k]: continue
    #    if len(np.unique(X[0:100, k])) > 2: continue
    #    if len(np.unique(X[:, k])) == 2: binary[k] = True

    for k in rng.randint(0, K, num_constraints):
        if binary[k]:
            if rng.rand() < 0.5:
                constraints[k] = Domain(0.0, 0.5)
            else:
                constraints[k] = Domain(0.5, 1.0)
        else:
            c = constraints[k]
            split = c.lo + rng.rand() * (c.hi - c.lo)
            try:
                l, r = constraints[k].split(split)
            except:
                continue
            constraints[k] = l if rng.rand() < 0.5 else r
            #print(f"{c} -> {constraints[k]}")

    return constraints


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
        at0 = dataset.at

        constraints = generate_random_constraints(dataset.X, 120, prune_seed)
        #print(constraints)
        at = at0.prune(constraints)
        #print(at)
        #at = veritas.AddTree(at, 0, 10)
        #print(at)
        print("prune:", at0.num_nodes(), "to", at.num_nodes(), "nodes")

        if algos[0] == "1":
            print("\n== VERITAS ======================================")
            search = NodeSearch(at)
            #search.set_eps(1.0) # disable ARA*
            done = search.step_for(MAX_TIME)
            print(done, search.num_solutions(), search.current_bound())
            if search.num_solutions() > 0:
                s = search.get_solution(0)
                print("solution", s.output, s.time)
#
#                b = s.box()
#                p = np.zeros((1, len(dataset.meta["columns"])))
#                for i, d in b.items(): p[0, i] = sum(map(np.nan_to_num, d))/2
#
#                print(b)
#                print(p, "->", at.eval(p))
            result["veritas"] = {}
            result["veritas"]["bounds"] = [s.bound for s in search.stats.snapshots]
            result["veritas"]["times"] = [s.time for s in search.stats.snapshots]
            #result["veritas"]["num_steps"] = [s.num_steps for s in search.stats.snapshots]

            result["veritas"]["solutions"] = []
            for i in range(search.num_solutions()):
                s = search.get_solution(i)
                result["veritas"]["solutions"].append(
                        {"output": s.output, "time": s.time, "eps": s.eps})

        if algos[1] == "1":
            print("\n== VERITAS0 =====================================")
            atcpy = _veritas_at_to_veritas0_at(at)
            opt = veritas0.Optimizer(maximize=atcpy, max_memory=MAX_MEM)
            #opt.prune_box(box, instance=1)
            dur, oom = opt.astar(max_time=MAX_TIME)
            result["veritas0"] = opt.stats()
            result["veritas0_time"] = dur
            result["veritas0_oom"] = oom
            print("   ", result["veritas0"]["bounds"][-1], dur)
            if len(result["veritas0"]["solutions"]) > 0:
                print("    sol:", result["veritas0"]["solutions"][0])
            print("    veritas0 time", dur)

            print("\n== VERITAS ARA* =================================")
            opt = veritas0.Optimizer(maximize=atcpy, max_memory=MAX_MEM)
            #opt.prune_box(box, instance=1)
            dur, oom = opt.arastar(max_time=MAX_TIME)
            result["veritas0_ara"] = opt.stats()
            result["veritas0_ara_time"] = dur
            result["veritas0_ara_oom"] = oom
            print("   ", result["veritas0_ara"]["bounds"][-1], dur)
            if len(result["veritas0_ara"]["solutions"]) > 0:
                print("    sol:", max(result["veritas0_ara"]["solutions"], key=lambda s: s[1])[1])
            print("    veritas0 time", dur)

        if algos[2] == "1":
            print("\n== MERGE ========================================")
            opt = Optimizer(maximize=at, max_memory=MAX_MEM)
            opt.prune_box(box, instance=1)
            data = opt.merge(max_time=MAX_TIME)
            result["merge"] = data
            print("    merge time", data["total_time"])

        if algos[3] == "1":
            print("\n== KANTCHELIAN MIPS =============================")
            kan = KantchelianOutputOpt(at, max_time=MAX_TIME)
            #kan.constraint_to_box(constraints)
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
    assert len(algos) == 4
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
