import os, sys, json, gzip, time
import datasets
from veritas.robustness import RobustnessSearch, VeritasRobustnessSearch, MilpRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np

def write_result(result, outfile):
    result_str = json.dumps(result)
    result_bytes = result_str.encode('utf-8')  
    outfile.write(result_bytes)
    outfile.write(b"\n")

    # Make sure we write so we don't lose anything on error
    outfile.flush()
    os.fsync(outfile)

def run(at0, at1, example, start_delta, max_time, algos, result):
    if algos[0] == "1":
        print("\n== VERITAS ======================================", f"({time.ctime()})")
        ver = VeritasRobustnessSearch(at0, at1, example, start_delta=start_delta,
                max_time=max_time,
                stop_condition=RobustnessSearch.NO_STOP_COND)
        ver_norm, ver_lo, ver_hi = ver.search()
        result["veritas_deltas"] = ver.delta_log
        result["veritas_log"] = ver.log
        result["veritas_time"] = ver.total_time
        result["veritas_time_p"] = ver.total_time_p
        result["veritas_examples"] = ver.generated_examples
        print("veritas time", ver.total_time, ver.total_time_p)

    if algos[1] == "1":
        print("\n== KANTCHELIAN MILP =============================", f"({time.ctime()})")
        kan = KantchelianTargetedAttack(at0, at1, example=example)
        kan.optimize()
        kan_example, kan_pred0, kan_pred1, kan_norm = kan.solution()
        result["kantchelian"] = kan.stats()
        result["kantchelian_example"] = kan_example
        result["kantchelian_pred"] = (kan_pred0, kan_pred1)
        result["kantchelian_delta"] = kan_norm
        print("kantchelian time", kan.total_time, kan.total_time_p)

    if algos[2] == "1":
        print("\n== MILP BINARY SEARCH ===========================", f"({time.ctime()})")
        milp = MilpRobustnessSearch(at0, at1, example, start_delta=start_delta,
                max_time=max_time, stop_condition=RobustnessSearch.NO_STOP_COND)
        ver_norm, ver_lo, ver_hi = milp.search()
        result["milp_deltas"] = milp.delta_log
        result["milp_time"] = milp.total_time
        result["milp_time_p"] = milp.total_time_p
        result["milp_examples"] = milp.generated_examples
        print("MILP BIN SEARCH time", milp.total_time, milp.total_time_p)

def robustness_experiment_multiclass(dataset, example_is, max_time,
        start_delta, num_classes, outfile, algos):
    for example_i in example_is:
        example = list(dataset.X.iloc[example_i,:])
        example_label = int(dataset.y[example_i])
        at0 = dataset.at[example_label]
        for target_label in [j for j in range(num_classes) if j!=example_label]:
            at1 = dataset.at[target_label]

            print(f"\n\n== EXAMPLE {example_i}: {example_label} vs {target_label} ({algos}) ===========")
            result = {
                "max_time": max_time,
                "example_i": example_i,
                "example_label": example_label,
                "target_label": target_label,
                "algos": algos,
            }
            run(at0, at1, example, start_delta, max_time, algos, result)
            write_result(result, outfile)

def robustness_experiment_binary(dataset, example_is, max_time, start_delta,
        outfile, algos):
    for example_i in example_is:
        example = list(dataset.X.iloc[example_i,:])
        example_np = np.array([example])
        example_label = int(dataset.y[example_i])
        at = dataset.at
        predicted_value = at.eval(example_np)[0]

        print("example", example)
        print("misclassified?", not (example_label == (predicted_value >= 0.0)))
        if example_label == 1:  # label is POS, MINIMIZE at output
            print("POS:", predicted_value, "with true label", example_label)
            at0, at1 = at, None
        else:                   # label is NEG, MAXIMIZE at output
            print("NEG:", predicted_value, "with true label", example_label)
            at0, at1 = None, at

        print(f"\n\n== EXAMPLE {example_i}: {example_label} ({algos}) ===========")
        result = {
            "max_time": max_time,
            "example_i": example_i,
            "example_label": example_label,
            "target_label": abs(example_label-1),
            "algos": algos,
        }
        run(at0, at1, example, start_delta, max_time, algos, result)
        write_result(result, outfile)

def parse_dataset(dataset):
    if dataset == "covtype":
        dataset = datasets.CovtypeNormalized() # normalized
        dataset.load_dataset()
        dataset.load_model(80, 8)
        start_delta = 0.2
        num_classes = 2
        T, L = 2, 2 # from 2, 3, too slow!
    elif dataset == "covtype_small":
        dataset = datasets.CovtypeNormalized() # normalized
        dataset.load_dataset()
        dataset.load_model(20, 4)
        start_delta = 0.2
        num_classes = 2
        T, L = 2, 2 # from 2, 3, too slow!
    elif dataset == "f-mnist":
        dataset = datasets.FashionMnist() # non-normalized (0-255)
        dataset.load_dataset()
        dataset.load_model(200, 8)
        start_delta = 20
        num_classes = 10
        T, L = 2, 1
    elif dataset == "mnist":
        dataset = datasets.Mnist() # non-normalized (0-255)
        dataset.load_dataset()
        dataset.load_model(200, 8)
        start_delta = 40
        num_classes = 10
        T, L = 2, 2
    elif dataset == "mnist_small":
        dataset = datasets.Mnist() # non-normalized (0-255)
        dataset.load_dataset()
        dataset.load_model(20, 4)
        start_delta = 40
        num_classes = 10
        T, L = 2, 2
    elif dataset == "higgs":
        dataset = datasets.LargeHiggs() # normalized
        dataset.load_dataset()
        dataset.load_model(300, 8)
        start_delta = 0.05
        num_classes = 2
        T, L = 4, 1
    elif dataset == "ijcnn1":
        dataset = datasets.Ijcnn1() # normalized
        dataset.load_dataset()
        dataset.load_model(60, 8)
        start_delta = 0.1
        num_classes = 2
        T, L = 2, 2
    elif dataset == "webspam":
        dataset = datasets.Webspam() # normalized
        dataset.load_dataset()
        dataset.load_model(100, 8)
        start_delta = 0.05
        num_classes = 2
        T, L = 2, 1
    elif dataset == "mnist2v6":      # non-normalized! (0-255)
        dataset = datasets.Mnist2v6()
        dataset.load_dataset()
        dataset.load_model(1000, 4)
        start_delta = 40
        num_classes = 2
        T, L = 4, 1
    else:
        raise ValueError("invalid dataset")

    return dataset, start_delta, num_classes, T, L

def main():
    dataset = sys.argv[1]
    example_is = range(*(int(i) for i in sys.argv[2].split(":")))
    outfile_base = sys.argv[3]
    max_time = int(sys.argv[4])
    algos = sys.argv[5] # algo order: veritas merge treeck kantchelian
    assert len(algos) == 3
    outfile = f"{outfile_base}-{dataset}-time{max_time}-{example_is.start}:{example_is.stop}-{algos}.gz"

    dataset, start_delta, num_classes, T, L = parse_dataset(dataset)

    if "--yes" not in sys.argv and os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        try:
            if num_classes == 2:
                robustness_experiment_binary(dataset, example_is, max_time,
                        start_delta, f, algos)
            else:
                robustness_experiment_multiclass(dataset, example_is, max_time,
                        start_delta, num_classes, f, algos)
        finally: 
            print("results written to", outfile, f"({time.ctime()})")

if __name__ == "__main__":
    main()
