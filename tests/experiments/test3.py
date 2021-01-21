import datasets
from veritas import Optimizer
from robust import RobustnessSearch, VeritasRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.Mnist()
mnist.load_model(10, 4)
mnist.load_dataset()
example_i = 1
example = list(mnist.X.iloc[example_i,:])
example_label = int(mnist.y[example_i])

at0 = mnist.at[example_label]
at1 = mnist.at[(example_label+1)%10]

actual_prediction0 = at0.predict_single(example)
actual_prediction1 = at1.predict_single(example)

opt = Optimizer(minimize=mnist.at[1], maximize=mnist.at[0],
        matches=set(), match_is_reuse=False)
opt.prune_example(example, delta=20)

data = opt.merge(max_merge_depth=2)
print(data)
