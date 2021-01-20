import sys
import json
import numpy as np
from scale import MnistXvallPrunedScaleExperiment, mnist_robust_search

import milp.xgbKantchelianAttack as chen
from veritas.kantchelian import MILP

import matplotlib.pyplot as plt


outfile = "/tmp/test_outfile"
seed = 0
start_delta = 20.1

exp = MnistXvallPrunedScaleExperiment(max_memory=4*1024*1024*1024, max_time=10, num_threads=1)
exp.steps_kwargs["min_output_difference"] = 0.0
#exp.confirm_write_results(outfile)
exp.do_merge = False

example_i = 12
exp.load_example(example_i, start_delta)
exp.load_model(num_trees=10, depth=4, label=exp.example_label)
#mnist_robust_search(outfile, exp, example_i, 0, True, start_delta, seed, num_trees=10, tree_depth=4)

actual_prediction = exp.at.predict_single(exp.example)
print("actual_prediction", actual_prediction)

#print(exp.at)
m = MILP(exp.at, target_label=0 if actual_prediction>0.0 else 1, example=exp.example)
m.optimize()
adv_example, adv_prediction, norm = m.solution()

print("adv", list(adv_example), "norm", norm)
print("adv prediction      ", adv_prediction, actual_prediction)
print("adv prediction check", exp.at.predict_single(adv_example))
print("norm check", max(abs(adv_example-exp.example)))

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(exp.example.to_numpy().reshape((28,28)))
ax1.imshow(adv_example.to_numpy().reshape((28,28)))
ax2.imshow((exp.example-adv_example).to_numpy().reshape((28,28)))
plt.show()

#print("test", m._get_objective_weights([1.0, 5.0, 8.0], 0.0))
#print("test", m._get_objective_weights([1.0, 5.0, 8.0], 3.0))
#print("test", m._get_objective_weights([1.0, 5.0, 8.0], 6.0))
#print("test", m._get_objective_weights([1.0, 5.0, 8.0], 8.0))
#print("test", m._get_objective_weights([1.0, 5.0, 8.0], 9.0))


#exp.write_results()

#chen_model = chen.xgboost_wrapper(exp.model, binary=True)
#guard_val = chen.GUARD_VAL
#round_digits = chen.ROUND_DIGITS
#attack = chen.xgbKantchelianAttack(chen_model, guard_val=guard_val, round_digits=round_digits)

#print(exp.at)
#for x in chen_model.json_file:
#    print(json.dumps(x, indent=2))

#adv = attack.attack(exp.example, exp.example_label)
#dist = np.max(np.abs(adv-exp.example))
#
#print(attack.m.display())
#
#print('adv:', adv)
#print('dist:', dist)
#print('original label:', exp.example_label)
