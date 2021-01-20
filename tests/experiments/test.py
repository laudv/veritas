import sys
import json
import numpy as np
from scale import MnistXvallPrunedScaleExperiment, mnist_robust_search

import milp.xgbKantchelianAttack as chen
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack

import matplotlib.pyplot as plt

outfile = "/tmp/test_outfile"
seed = 0
start_delta = 20.1

exp = MnistXvallPrunedScaleExperiment(max_memory=4*1024*1024*1024, max_time=10, num_threads=1)
exp.steps_kwargs["min_output_difference"] = 0.0
#exp.confirm_write_results(outfile)
exp.do_merge = False

example_i = 16
exp.load_example(example_i, start_delta)
target_label = (exp.example_label + 1) % 10
exp.load_model(num_trees=10, depth=4, label=target_label)
exp.target_at = exp.at
exp.load_model(num_trees=10, depth=4, label=exp.example_label)
#mnist_robust_search(outfile, exp, example_i, 0, True, start_delta, seed)

actual_prediction0 = exp.at.predict_single(exp.example)
actual_prediction1 = exp.target_at.predict_single(exp.example)

#m = KantchelianAttack(exp.target_at, target_output=True, example=exp.example)
#m.optimize()
#adv_example, adv_prediction, norm = m.solution()
#adv_prediction0 = adv_prediction
#adv_prediction1 = adv_prediction

m = KantchelianTargetedAttack(exp.at, exp.target_at, example=exp.example)
m.optimize()
adv_example, adv_prediction0, adv_prediction1, norm = m.solution()

print("act prediction      ", actual_prediction0, actual_prediction1)
print("adv prediction      ", adv_prediction0, adv_prediction1)
print("adv prediction check", exp.at.predict_single(adv_example), exp.target_at.predict_single(adv_example))
print("reported norm", norm)
print("norm check   ", max(abs(adv_example-exp.example)))

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(exp.example.to_numpy().reshape((28,28)))
ax1.imshow(adv_example.to_numpy().reshape((28,28)))
ax2.imshow((exp.example-adv_example).to_numpy().reshape((28,28)))
plt.show()
