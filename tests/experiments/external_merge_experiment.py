import sys, os, gzip, time
from robustness_experiment import parse_dataset, write_result
from external_merge import external_merge_binary, external_merge_multiclass

def main():
    dataset = sys.argv[1]
    example_is = range(*(int(i) for i in sys.argv[2].split(":")))
    outfile_base = sys.argv[3]
    outfile = f"{outfile_base}-{dataset}-{example_is.start}:{example_is.stop}-EXT.gz"

    dataset, start_delta, num_classes, T, L = parse_dataset(dataset)

    if "--yes" not in sys.argv and os.path.isfile(outfile):
        if input(f"override {outfile}? ") != "y":
            print("OK BYE")
            sys.exit()

    with gzip.open(outfile, "wb") as f:
        try:
            if num_classes == 2:
                print("\n== MERGE (external) =============================", f"({time.ctime()})")
                example_is, example_labels, deltas, times = external_merge_binary(
                        dataset, example_is, start_delta, T, L)
                for i, el, ds, ts in zip(example_is, example_labels, deltas, times):
                    result = {
                        "example_i": i,
                        "example_label": el,
                        "merge_ext": {
                            "deltas": ds,
                            "times": ts,
                            "max_clique": T,
                            "max_level": L
                        }
                    }
                    write_result(result, f)
            else:
                print("\n== MERGE (external) multiclass ==================", f"({time.ctime()})")
                example_is, example_labels, target_labels, deltas, times = external_merge_multiclass(
                        dataset, example_is, start_delta, T, L, num_classes)
                print("DEBUG", example_is, example_labels, target_labels)
                for i, el, tl, ds, ts in zip(example_is, example_labels,
                        target_labels, deltas, times):
                    result = {
                        "example_i": i,
                        "example_label": el,
                        "target_label": tl,
                        "merge_ext": {
                            "deltas": ds,
                            "times": ts,
                            "max_clique": T,
                            "max_level": L
                        }
                    }
                    write_result(result, f)
        finally: 
            print("results written to", outfile, f"({time.ctime()})")

if __name__ == "__main__":
    main()
