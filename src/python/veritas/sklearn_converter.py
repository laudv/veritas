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

class SklRfAddTreeConverter(AddTreeConverter):
    def convert(self, ensemble, silent):
        try:
            from sklearn.ensemble import \
                    RandomForestClassifier, \
                    RandomForestRegressor
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("no sklearn")

        num_leaf_values = 1

        # TODO add sklearn boosted trees, extra trees, isolation forest, ...

        if isinstance(ensemble, RandomForestRegressor):
            at_type = AddTreeType.REGR_MEAN 

            num_leaf_values = ensemble.n_outputs_ 
            def extract_value_fun(v, i):
                return v[i][0]

            if not silent:
                print(f"SKLEARN: RF regressor with {num_leaf_values} target(s)")

        elif isinstance(ensemble, RandomForestClassifier):
            at_type = AddTreeType.CLF_MEAN
            num_leaf_values = ensemble.n_classes_ if ensemble.n_classes_ > 2 else 1 
            if num_leaf_values > 2:
                def extract_value_fun(v, i):
                    return v[0][i]/sum(v[0])
            else:
                def extract_value_fun(v, i):
                    assert i == 0
                    return v[0][1]/sum(v[0])

            if not silent:
                print(f"SKLEARN: RF classifier with {num_leaf_values} classes")

        else:
            raise InapplicableAddTreeConverter(f"not sklearn rf: {type(ensemble)}")

        at = AddTree(num_leaf_values, at_type)
        for tree in ensemble.estimators_:
            addtree_sklearn_tree(at, tree.tree_, extract_value_fun)

        if at_type != AddTreeType.REGR_MEAN:
            for k in range(num_leaf_values):
                at.set_base_score(k, -len(at)/2.0)

        return at

class SklGbdtAddTreeConverter(AddTreeConverter):
    def convert(self, ensemble, silent):
        try:
            from sklearn.ensemble import \
                    GradientBoostingClassifier, \
                    GradientBoostingRegressor
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("no sklearn")

        if isinstance(ensemble, GradientBoostingRegressor):
            at_type = AddTreeType.REGR
            num_leaf_values = 1
            base_scores = [ensemble.init_.constant_[0][k] for k in range(num_leaf_values)]
            def extract_value_fun(v, i):
                return v[i][0] * ensemble.learning_rate

            if not silent:
                print(f"SKLEARN: GDBT regressor with {num_leaf_values} target(s)")

        elif isinstance(ensemble, GradientBoostingClassifier):
            # __import__('pprint').pprint(ensemble.__dict__)
            test_instance = np.zeros((1, ensemble.n_features_in_))
            test_instance_pred = ensemble.init_.predict_proba(test_instance)
            size_pred = test_instance_pred.shape[1]

            if size_pred == 2:
                num_leaf_values = 1
                base_scores = ensemble._loss.link.link(test_instance_pred[:,1]).ravel()
            else:
                num_leaf_values = size_pred
                base_scores = ensemble._loss.link.link(test_instance_pred).ravel()

            at_type = AddTreeType.CLF_SOFTMAX

            def extract_value_fun(v, i):
                return v[i][0] * ensemble.learning_rate

            if not silent:
                print(f"SKLEARN: GDBT regressor with {num_leaf_values} target(s)")

        else:
            raise InapplicableAddTreeConverter(f"not sklearn gbdt: {type(ensemble)}")

        at = AddTree(num_leaf_values, at_type)
        for trees in ensemble.estimators_:
            for k, tree in enumerate(trees):
                #import sklearn.tree
                #print(sklearn.tree.export_text(tree))

                at_tmp = AddTree(1, at_type)
                addtree_sklearn_tree(at_tmp, tree.tree_, extract_value_fun)
                at.add_trees(at_tmp, k)

        for k, val in enumerate(base_scores):
            at.set_base_score(k, val)

        return at


class SklTreeAddTreeConverter(AddTreeConverter):
    def convert(self, ensemble, silent):
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("no sklearn")

        if isinstance(ensemble, DecisionTreeClassifier):
            tree = ensemble
            at_type = AddTreeType.CLF_MEAN
            num_leaf_values = 1
            def extract_value_fun(v, i):
                assert i == 0
                return v[0][1]/sum(v[0])

            if not silent:
                print(f"SKLEARN: single tree classifier with {num_leaf_values} classes")

        elif isinstance(ensemble, DecisionTreeRegressor):
            tree = ensemble
            at_type = AddTreeType.REGR_MEAN
            num_leaf_values = 1

            def extract_value_fun(v, i):
                assert i == 0
                return v[i][0]

        else:
            raise InapplicableAddTreeConverter(f"not sklearn clf tree: {type(ensemble)}")

        at = AddTree(num_leaf_values, at_type)
        addtree_sklearn_tree(at, tree, extract_value_fun)
        return at


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
            #split_value = thrs
            #split_value = np.nextafter(np.float32(
            #    thrs), np.float32(np.inf))  # <= splits
            split_value = np.nextafter(np.float64(
                thrs), np.float64(np.inf))  # <= splits
            t.split(m, feat_id, split_value)
            stack.append((tree.children_right[n], t.right(m)))
            stack.append((tree.children_left[n], t.left(m)))
        else:
            for i in range(at.num_leaf_values()):
                leaf_value = extract_value_fun(tree.value[n], i)
                t.set_leaf_value(m, i, leaf_value)
