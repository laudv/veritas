import os, json
import subprocess, resource

from sklearn.datasets import dump_svmlight_file

MERGE_EXECUTABLE = os.path.join(os.environ["HOME"], "repos", "treeVerification", "treeVerify")
MERGE_CRAP_DIR = "/tmp/merge_crap"
INPUT_FILE = os.path.join(MERGE_CRAP_DIR, f"inputs-{os.getpid()}.libsvm")
MODEL_FILE = os.path.join(MERGE_CRAP_DIR, f"model-{os.getpid()}.json")
CONFIG_FILE = os.path.join(MERGE_CRAP_DIR, f"config-{os.getpid()}.json")

def prepare_xgb_model_json(xgbjson, columns, negate_leaf=False):
    out = []
    fmap = { n: i for i, n in enumerate(columns) }
    for j in xgbjson:
        out.append(json.loads(j))
        stack = [out[-1]]
        while len(stack) > 0:
            n = stack.pop()
            if "split" in n:
                n["split"] = fmap[n["split"]]
                children = { child["nodeid"]: child for child in n["children"] }
                left_id = n["yes"]
                right_id = n["no"]
                stack += [children[right_id], children[left_id]]
            elif negate_leaf and "leaf" in n:
                #print("neg leaf", n["leaf"])
                n["leaf"] = -n["leaf"]
    return out

def prepare_xgb_model_json_multiclass(xgbjson, columns, label, target_label, num_classes):
    xgbjson0 = [xgbjson[i] for i in range(label, len(xgbjson), num_classes)]
    xgbjson1 = [xgbjson[i] for i in range(target_label, len(xgbjson), num_classes)]
    print("num_trees", len(xgbjson0), len(xgbjson1))
    return prepare_xgb_model_json(xgbjson1, columns, negate_leaf=True)\
        + prepare_xgb_model_json(xgbjson0, columns, negate_leaf=False)

def run_process(max_mem=8*1024*1024*1024, max_time=60*60*24):
    # https://gist.github.com/s3rvac/f97d6cbdfdb15c0a32e7e941f7f4a3fa
    def limit_virtual_memory():
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, resource.RLIM_INFINITY))

    p = subprocess.Popen(
        [MERGE_EXECUTABLE, CONFIG_FILE],
        stdout=subprocess.PIPE,
        preexec_fn=limit_virtual_memory)
    exception = False
    try:
        outs, errs = p.communicate(timeout=max_time)
    except Exception as e:
        print("merge out of time / memory")
        print("-- exception:", e)
        with open(f"/proc/{p.pid}/statm", "r") as f:
            print("statm", f.read())
        p.kill()
        outs, errs = p.communicate() # clean buffers
        exception=True

    #print("\n=====\n", outs.decode("utf-8"), "\n=====\n", sep="")
    return outs.decode("utf-8"), exception

def process_merge_output(out):
    all_times = []
    all_deltas = []
    times = []
    deltas = []
    for line in out.split("\n"):
        if line.startswith("Can model be guaranteed robust within eps"):
            delta = float(line[42:line.find("?")])
            is_robust_for_delta = bool(int(line[-1]))
            if len(deltas) == 0: # add the first delta, which is not announced with "next eps:..."
                deltas.append(delta)
            print(f"External merge: is robust for {delta}? {is_robust_for_delta}")
        elif line.startswith("**************** this eps ends, next eps:"):
            delta = float(line[41:line.find(" *********************")])
            deltas.append(delta)
        elif line.startswith("time="):
            t = float(line[5:])
            times.append(t)
        elif line.startswith("=================start index:0, num of points:"):
            s0 = line.find("current index:") + 14
            s1 = line.find(",", s0)
            t = line[s0:s1]
            print(line, "->", f"'{t}'")
            if len(times) > 0:
                all_deltas.append(deltas)
                all_times.append(times)
            times = []
            deltas = []
    if len(times) > 0:
        all_deltas.append(deltas)
        all_times.append(times)
    return all_deltas, all_times

def write_merge_config(start_delta, num_examples, max_clique, max_level):
    if not os.path.isdir(MERGE_CRAP_DIR):
        os.makedirs(MERGE_CRAP_DIR)
    config = f"""
{{
    "inputs":       "{INPUT_FILE}",
    "model":        "{MODEL_FILE}",
    "start_idx":    0,
    "num_attack":   {num_examples},
    "eps_init":     {start_delta},
    "max_clique":   {max_clique},
    "max_search":   10,
    "max_level":    {max_level},
    "num_classes":  2,
    "feature_start": 0
}}
"""
    with open(CONFIG_FILE, "w") as f:
        f.write(config)

def write_inputs(examples, labels):
    with open(INPUT_FILE, "w") as f:
        # somehow struggles with sparse svm format; let's write it all out
        for example, label in zip(examples, labels):
            vs = ' '.join(f"{i}:{v}" for i, v in enumerate(example))
            print(f"{label} {vs}", file=f)
            #dump_svmlight_file([example], [label], f, zero_based=True)

def clear_crap():
    os.remove(INPUT_FILE)
    os.remove(MODEL_FILE)
    os.remove(CONFIG_FILE)

def external_merge_binary(dataset, example_is, start_delta,
        max_clique, max_level):
    xgbmodel = dataset.model
    columns = dataset.meta["columns"]
    examples = [list(dataset.X.iloc[example_i,:]) for example_i in example_is]
    example_labels = [int(dataset.y[example_i]) for example_i in example_is]

    write_merge_config(start_delta, len(examples), max_clique, max_level)

    with open(MODEL_FILE, "w") as f:
        dump = xgbmodel.get_dump(dump_format="json")
        dump = prepare_xgb_model_json(dump, columns)
        print("num_trees", len(dump))
        json.dump(dump, f)

    write_inputs(examples, example_labels)

    out, exception = run_process()
    deltas, times = process_merge_output(out)
    clear_crap()
    return deltas, times, exception

def external_merge_multiclass(dataset, example_is, start_delta, max_clique,
        max_level, num_classes):
    xgbmodel = dataset.model
    columns = dataset.meta["columns"]
    examples = [list(dataset.X.iloc[example_i,:]) for example_i in example_is]
    example_labels = [int(dataset.y[example_i]) for example_i in example_is]

    write_merge_config(start_delta, len(examples), max_clique, max_level)

    def write_model(label, target_label):
        with open(MODEL_FILE, "w") as f:
            dump = xgbmodel.get_dump(dump_format="json")
            dump = prepare_xgb_model_json_multiclass(dump, columns, label,
                    target_label, num_classes)
            json.dump(dump, f)

    permutation = []
    deltas, times = [], []

    for l0 in range(num_classes):
        ex_filtered = [ex for ex, l in zip(examples, example_labels) if l==l0]
        ex_is = [i for i, l in enumerate(example_labels) if l==l0]
        if len(ex_filtered) == 0: continue
        for l1 in [l for l in range(num_classes) if l != l0]:
            print(f"{l0} -> {l1} ({len(ex_filtered)}, {ex_is})")
            write_model(l0, l1)

            permutation += [(i, l1) for i in ex_is]

            write_inputs(ex_filtered, [1.0]*len(ex_filtered))

            out, exception = run_process()
            ds, ts = process_merge_output(out)

            deltas += ds
            times += ts

            if exception: raise RuntimeError("MERGE CRASHED")


    # SORT deltas, times
    x = sorted(sorted(zip(permutation, deltas, times),
        key=lambda p: p[0][1]), key=lambda p: p[0][0])

    permutation, deltas, times = zip(*x)
    print(permutation, len(permutation))
    example_is, target_labels = zip(*permutation)
    print("DEBUG2", example_is, target_labels)

    clear_crap()
    return example_is, target_labels, deltas, times
