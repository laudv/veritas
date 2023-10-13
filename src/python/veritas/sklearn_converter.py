# \file sklearn.py
#
# This requires `scikit-learn` to be installed.
#
# Copyright 2023 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos, Alexander Schoeters

import numpy as np

from . import AddTree, AddTreeType, AddTreeConverter
from . import InapplicableAddTreeConverter

class SklAddTreeConverter(AddTreeConverter):
    def convert(self, model):
        return addtree_sklearn_ensemble(model)

def addtree_sklearn_tree(at, tree, extract_value_fun):
    import sklearn.tree as sktree
    if isinstance(tree, sktree.DecisionTreeClassifier) \
            or isinstance(tree, sktree.DecisionTreeRegressor):
        tree = tree.tree_

    if not isinstance(tree, sktree._tree.Tree):
        raise InapplicableAddTreeConverter("not a sklearn Tree")

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
    try:
        from sklearn.ensemble import \
                RandomForestClassifier, \
                RandomForestRegressor
    except ModuleNotFoundError:
        raise InapplicableAddTreeConverter("not a sklearn model")

    num_leaf_values = 1

    # TODO add sklearn boosted trees, extra trees, isolation forest, ...

    if isinstance(ensemble, RandomForestRegressor):
        print("SKLEARN: regressor")
        at_type = AddTreeType.RF_REGR

        def extract_value_fun(v, i):
            # print("skl leaf regr", v)
            return v[0]

    elif isinstance(ensemble, RandomForestClassifier):
        num_leaf_values = ensemble.n_classes_ if ensemble.n_classes_  > 2 else 1 
        at_type = AddTreeType.RF_BINARY
        if num_leaf_values > 2:
            at_type = AddTreeType.RF_MULTI
        print(f"SKLEARN: classifier with {num_leaf_values} classes")

        def extract_value_fun(v, i):
            # print("skl leaf clf", v[0], sum(v[0]), v[0][i])
            return v[0][i]/sum(v[0])
    else:
        raise InapplicableAddTreeConverter(f"not sklearn: {type(ensemble)}")

    at = AddTree(num_leaf_values, at_type)
    for tree in ensemble.estimators_:
        addtree_sklearn_tree(at, tree.tree_, extract_value_fun)
    return at
