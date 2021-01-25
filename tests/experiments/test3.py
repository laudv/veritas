import datasets, sys
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.Higgs()
mnist.load_dataset()
mnist.load_model(10, 4)

plt.hist(mnist.y, bins=100)
plt.show()
plt.hist(np.log(mnist.y), bins=100)
plt.show()
sys.exit()
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
