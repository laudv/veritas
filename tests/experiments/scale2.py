import os, pickle
import numpy as np
import matplotlib.pyplot as plt

import treeck
from treeck.xgb import addtree_from_xgb_model

RESULT_DIR = "tests/experiments/scale"
model_name = "model-200.xgb"
print(f"loading model from file: {model_name}")
with open(os.path.join(RESULT_DIR, model_name), "rb") as f:
    model = pickle.load(f)

at = addtree_from_xgb_model(model)
at.base_score = 0

def visit_tree(tree, max_err):
    count = 0
    leaf_values = []
    candidates = []
    stack = [tree.root()]
    while len(stack) > 0:
        node = stack.pop()
        if tree.is_leaf(node):
            v = tree.get_leaf_value(node)
            leaf_values.append(v)
            continue
        elif tree.is_leaf(tree.left(node)) and tree.is_leaf(tree.right(node)):
            lv = tree.get_leaf_value(tree.left(node))
            rv = tree.get_leaf_value(tree.right(node))
            mv = (lv + rv) / 2.0
            merr = max(abs(lv-mv), abs(rv-mv))
            candidates.append((lv, rv))
            #if merr <= max_err:
            count += 1
                #print("  simplify", lv, rv, max(abs(lv-mv), abs(rv-mv)))
        stack.append(tree.right(node))
        stack.append(tree.left(node))

    return leaf_values, candidates, count




count_sum = 0
all_leaf_values = []
all_candidates = []
for i in range(len(at)):
    leaf_values, candidates, count = visit_tree(at[i], 0.01)
    count_sum += count
    #print(i, np.min(leaf_values), np.mean(leaf_values), np.max(leaf_values), np.std(leaf_values))
    all_leaf_values += leaf_values
    all_candidates += candidates

all_candidates_s = sorted(all_candidates, key=lambda t: abs(t[0]-t[1]))
print(all_candidates_s[:10])

#plt.hist(all_leaf_values, bins=100)
#plt.axvline(-0.01, ls=":", lw=1, c="gray")
#plt.axvline(0.01, ls=":", lw=1, c="gray")
#plt.show()



print(len(all_leaf_values), count_sum)
