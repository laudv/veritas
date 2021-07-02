import sys
import datasets
from veritas0 import Optimizer
from veritas0 import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas0.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

from external_merge import external_merge_binary, external_merge_multiclass

test = "binary"

if test == "binary":
    dataset = datasets.CovtypeNormalized()
    dataset.load_dataset()
    dataset.load_model(80, 8)
    example_is = range(115,120)
    examples = [list(dataset.X.iloc[example_i,:]) for example_i in example_is]
    example_labels = [int(dataset.y[example_i]) for example_i in example_is]
    
    deltas, times, exception = external_merge_binary(dataset, example_is, 0.1,
            2, 2)
    
    print("deltas", deltas)
    print("times", times)
    print("exc", exception)


elif test == "multi":
    dataset = datasets.Mnist()
    dataset.load_dataset()
    dataset.load_model(20, 4)
    example_is = range(100, 102)

    num_classes = 10

    external_merge_multiclass(dataset, example_is, 20, 2, 2, num_classes)


    #results = {}
    #for l0 in range(num_classes):
    #    ex_filtered = [ex for ex, l in zip(examples, example_labels) if l==l0]
    #    ex_is = [i for i, l in enumerate(example_labels) if l==l0]
    #    if len(ex_filtered) == 0: continue
    #    for l1 in [l for l in range(num_classes) if l != l0]:
    #        print(f"{l0} -> {l1} ({len(ex_filtered)}, {ex_is})")
    #        deltas, times, exception = external_merge_multiclass(dataset.model,
    #                dataset.meta["columns"], ex_filtered, l0, l1, 20, 2, 2, num_classes=num_classes)

    #        print("deltas", deltas)
    #        print("times", times)
    #        print("exc", exception)

    #        for ii, dd, tt in zip(ex_is, deltas, times):
    #            results[(ii, l1)] = {
    #                "deltas": dd,
    #                "times": tt,
    #                "exc": exception
    #            }

    #print(results)
