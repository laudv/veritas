import os, json
import subprocess, resource

from sklearn.datasets import dump_svmlight_file

MERGE_EXECUTABLE = os.path.join(os.environ["HOME"], "repos", "treeVerification", "treeVerify")
MERGE_CRAP_DIR = "/tmp/merge_crap"

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

def run_process(config_file, max_mem=4*1024*1024*1024, max_time=10.0):
    # https://gist.github.com/s3rvac/f97d6cbdfdb15c0a32e7e941f7f4a3fa
    def limit_virtual_memory():
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

    p = subprocess.Popen(
        [MERGE_EXECUTABLE, config_file],
        stdout=subprocess.PIPE,
        preexec_fn=limit_virtual_memory
    )
    exception = False
    try:
        outs, errs = p.communicate(timeout=max_time)
    except:
        print("merge out of time / memory")
        p.kill()
        outs, errs = p.communicate() # clean buffers
        exception=True

    #print(outs.decode("utf-8"), errs)
    return outs.decode("utf-8"), exception

def process_merge_output(out):
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
    return deltas, times

def external_merge(xgbmodel, columns, example, label, target_label, start_delta, max_clique, max_level, num_classes=2):
    if not os.path.isdir(MERGE_CRAP_DIR):
        os.makedirs(MERGE_CRAP_DIR)
    input_file = os.path.join(MERGE_CRAP_DIR, "inputs.libsvm")
    model_file = os.path.join(MERGE_CRAP_DIR, "model.json")
    config_file = os.path.join(MERGE_CRAP_DIR, "config.json")
    config = f"""
{{
    "inputs":       "{input_file}",
    "model":        "{model_file}",
    "start_idx":    0,
    "num_attack":   1,
    "eps_init":     {start_delta},
    "max_clique":   {max_clique},
    "max_search":   10,
    "max_level":    {max_level},
    "num_classes":  2,
    "feature_start": 0
}}
"""

    with open(config_file, "w") as f:
        f.write(config)

    if xgbmodel is not None: # if none, reuse model at {model_file}
        with open(model_file, "w") as f:
            dump = xgbmodel.get_dump(dump_format="json")
            if target_label is not None:
                dump = prepare_xgb_model_json_multiclass(dump, columns, label,
                        target_label, num_classes)
            else:
                dump = prepare_xgb_model_json(dump, columns)
                print("num_trees", len(dump))
            json.dump(dump, f)
    with open(input_file, "w") as f:
        # somehow struggles with sparse svm format?? let's write it all out
        vs = ' '.join(f"{i}:{v}" for i, v in enumerate(example))
        print(f"{label} {vs}", file=f)
        #dump_svmlight_file([example], [label], f, zero_based=True)

    #out = subprocess.run([MERGE_EXECUTABLE, config_file])
    out, exception = run_process(config_file)
    #print("==========")
    #print(out)
    #print("==========\n\n")
    deltas, times = process_merge_output(out)
    return deltas, times, exception
