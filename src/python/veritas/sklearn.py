## \file sklearn.py
# Copyright 2020 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import numpy as np

from . import AddTree

# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def _addtree_from_sklearn_tree(at, tree, extract_value_fun):
    t = at.add_tree()
    stack = [(0, t.root())]
    while len(stack) != 0:
        n, m = stack.pop()
        is_internal = tree.children_left[n] != tree.children_right[n]

        if is_internal:
            feat_id = tree.feature[n]
            split_value = np.nextafter(np.float32(tree.threshold[n]), np.float32(np.inf)) # <= splits
            t.split(m, feat_id, split_value)
            stack.append((tree.children_right[n], t.right(m)))
            stack.append((tree.children_left[n], t.left(m)))
        else:
            leaf_value = extract_value_fun(tree.value[n])
            t.set_leaf_value(m, leaf_value)

## Extract a Veritas AddTree from a scikit learn ensemble model (e.g. random
# forest)
def addtree_from_sklearn_ensemble(ensemble, extract_value_fun=lambda v: v[0]):
    at = AddTree()
    for tree in ensemble.estimators_:
        _addtree_from_sklearn_tree(at, tree.tree_, extract_value_fun)
    return at
    
## Extract `num_classes` Veritas AddTrees from a multi-class scikit learn
# ensemble model (e.g. random forest)
def addtrees_from_multiclass_sklearn_ensemble(ensemble, num_classes):
    addtrees = []
    for i in range(num_classes):
        extract_value_fun = lambda v: v[0][i]
        at = addtree_from_sklearn_ensemble(ensemble, extract_value_fun)
        addtrees.append(at)
    return addtrees



