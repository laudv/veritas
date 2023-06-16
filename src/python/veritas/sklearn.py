## \file sklearn.py
#
# This requires `scikit-learn` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import numpy as np

from . import AddTree
import sklearn.tree as sktree

class RfAddTree(AddTree):
    def predict(self, X):
        return self.eval(X)

    def predict_proba(self, X):
        return self.eval(X)

# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def addtree_from_sklearn_tree(at, tree, extract_value_fun):
    if isinstance(tree, sktree.DecisionTreeClassifier):
        tree = tree.tree_
    elif isinstance(tree, sktree.DecisionTreeRegressor):
        tree = tree.tree_

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
            for i in range(at.num_leaf_values()):
                leaf_value = extract_value_fun(tree.value[n], i)
                t.set_leaf_value(m, i, leaf_value)

## Extract a Veritas AddTree from a scikit learn ensemble model (e.g. random
# forest)
# 
# For binary classification: leaf values are converted to class ratio (first
# class count / total leaf count) divided by the number of trees.
# As far as I can tell, this corresponds with sklearn's predict_proba
def addtree_from_sklearn_ensemble(ensemble):
    num_trees = len(ensemble.estimators_)
    num_leaf_values = ensemble.n_classes_

    if "Regressor" in type(ensemble).__name__:
        print("SKLEARN: regressor")
        def extract_value_fun(v, i):
            #print("skl leaf regr", v)
            return v[0]/num_trees
    elif "Classifier" in type(ensemble).__name__:
        print(f"SKLEARN: classifier with {num_leaf_values} classes")
        def extract_value_fun(v, i):
            #print("skl leaf clf", v[0], sum(v[0]), v[0][i])
            return v[0][i]/sum(v[0])/num_trees
    else:
        raise RuntimeError("cannot determine extract_value_fun for:",
                type(ensemble).__name__)

    at = RfAddTree(num_leaf_values)
    for tree in ensemble.estimators_:
        addtree_from_sklearn_tree(at, tree.tree_, extract_value_fun)
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



