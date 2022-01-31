## \file sklearn.py
#
# This requires `scikit-learn` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import numpy as np

from . import AddTree

class RfAddTree(AddTree):
    def predict_proba(self, X):
        return (self.eval(X) - self.base_score) / len(self)

# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def _addtree_from_sklearn_tree(at, tree, extract_value_fun):
    t = at.add_tree()
    stack = [(0, t.root())]
    while len(stack) != 0:
        n, m = stack.pop()
        is_internal = tree.children_left[n] != tree.children_right[n]

        if is_internal:
            feat_id = tree.feature[n]
            thrs = tree.threshold[n]
            split_value = np.nextafter(np.float32(thrs), np.float32(np.inf)) # <= splits
            t.split(m, feat_id, split_value)
            stack.append((tree.children_right[n], t.right(m)))
            stack.append((tree.children_left[n], t.left(m)))
        else:
            leaf_value = extract_value_fun(tree.value[n])
            t.set_leaf_value(m, leaf_value)

## Extract a Veritas AddTree from a scikit learn ensemble model (e.g. random
# forest)
# 
# For binary classification: leaf values are converted to class ratio (first
# class count / total leaf count) divided by the number of trees.
# As far as I can tell, this corresponds with sklearn's predict_proba
def addtree_from_sklearn_ensemble(ensemble, extract_value_fun=None):
    num_trees = len(ensemble.estimators_)
    if extract_value_fun is None:
        if "Regressor" in type(ensemble).__name__:
            print("SKLEARN: regressor")
            extract_value_fun = lambda v: v[0]
        elif "Classifier" in type(ensemble).__name__:
            print("SKLEARN: binary classifier")
            extract_value_fun = lambda v: (v[0][1]/sum(v[0])) # class ratio
        else:
            raise RuntimeError("cannot determine extract_value_fun for:",
                    type(ensemble).__name__)

    at = RfAddTree()
    for tree in ensemble.estimators_:
        _addtree_from_sklearn_tree(at, tree.tree_, extract_value_fun)
    at.base_score = -num_trees / 2
    return at
    
## Extract `num_classes` Veritas AddTrees from a multi-class scikit learn
# ensemble model (e.g. random forest)
# TODO remove num_classes argument, extract from ensemble (n_classes_?)
def addtrees_from_multiclass_sklearn_ensemble(ensemble, num_classes):
    addtrees = []
    num_trees = len(ensemble.estimators_)
    for i in range(num_classes):
        extract_value_fun = lambda v: (v[0][i]/sum(v[0]))/num_trees
        at = addtree_from_sklearn_ensemble(ensemble, extract_value_fun)
        addtrees.append(at)
    return addtrees



