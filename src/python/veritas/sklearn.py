# \file sklearn.py
#
# This requires `scikit-learn` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos
import numpy as np

from . import AddTree, AddTreeType, AddTreeConverter
import sklearn.tree as sktree
from sklearn.ensemble import _forest


class Sk_AddTreeConverter(AddTreeConverter):
    def get_addtree(self,model):
        return addtree_sklearn_ensemble(model)


def addtree_sklearn_tree(at, tree, extract_value_fun):
    if isinstance(tree, sktree.DecisionTreeClassifier) or isinstance(tree, sktree.DecisionTreeRegressor):
        tree = tree.tree_

    t = at.add_tree()
    stack = [(0, t.root())]
    while len(stack) != 0:
        n, m = stack.pop()
        is_internal = tree.children_left[n] != tree.children_right[n]

        if is_internal:
            feat_id = tree.feature[n]
            thrs = tree.threshold[n]
            split_value = np.nextafter(np.float32(
                thrs), np.float32(np.inf))  # <= splits
            t.split(m, feat_id, split_value)
            stack.append((tree.children_right[n], t.right(m)))
            stack.append((tree.children_left[n], t.left(m)))
        else:
            for i in range(at.num_leaf_values()):
                leaf_value = extract_value_fun(tree.value[n], i)
                t.set_leaf_value(m, i, leaf_value)


def addtree_sklearn_ensemble(ensemble):
    num_trees = len(ensemble.estimators_)
    num_leaf_values = 1

    if "Regressor" in type(ensemble).__name__:
        print("SKLEARN: regressor")
        type_ = AddTreeType.RF_REGR

        def extract_value_fun(v, i):
            # print("skl leaf regr", v)
            return v[0]
    elif "Classifier" in type(ensemble).__name__:
        num_leaf_values = ensemble.n_classes_ if ensemble.n_classes_  > 2 else 1 
        type_ = AddTreeType.RF_MULTI if num_leaf_values > 2 else AddTreeType.RF_CLF
        print(f"SKLEARN: classifier with {num_leaf_values} classes")

        def extract_value_fun(v, i):
            # print("skl leaf clf", v[0], sum(v[0]), v[0][i])
            return v[0][i]/sum(v[0])
    else:
        raise RuntimeError("cannot determine extract_value_fun for:",
                           type(ensemble).__name__)

    at = AddTree(num_leaf_values, type_)
    for tree in ensemble.estimators_:
        addtree_sklearn_tree(at, tree.tree_, extract_value_fun)
    return at
