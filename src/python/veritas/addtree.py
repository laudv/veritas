import json
import numpy as np

from xgboost.sklearn import XGBModel
from xgboost.core import Booster

import sklearn.tree as sktree
from sklearn.ensemble import _forest

from . import AddTree


class ConAddTree(AddTree):
    def predict(self, X):
        return self.eval(X)

    def predict_proba(self, X):
        return 1/(1+np.exp(-self.eval(X)))


def get_addtree(model):
    # Check model type

    # XGB
    if isinstance(model, XGBModel):
        # Multiclass
        if "num_class" in model.get_params():
            nclasses = model.get_params()["num_class"]
            return [addtree_xgb(model, multiclass=(clazz, nclasses), base_score=0.5) for clazz in range(nclasses)]
        else:
            # Regression or binary class
            return addtree_xgb(model)
    # Sklearn RandomForest
    elif str(getattr(model, '__module__', None)) == "sklearn.ensemble._forest":
        # Multiclass
        if isinstance(model, _forest.RandomForestClassifier) and model.n_classes_ > 1:
            pass
        else:
            # Regression or binary
            return addtree_sklearn_ensemble(model)
    # LGBM
    elif 1 == 1:
        pass

    return -1


def addtree_xgb(model, multiclass=(0, 1), base_score=0.0):
    base_score = model.base_score if model.base_score is not None else base_score
    model = model.get_booster()
    assert isinstance(model, Booster), f"not xgb.Booster but {type(model)}"

    dump = model.get_dump("", dump_format="json")
    at = ConAddTree(1)

    at.base_score = base_score
    offset, num_classes = multiclass

    for i in range(offset, len(dump), num_classes):
        _parse_tree(at, dump[i])

    return at


def feat2id_map(f): return int(f[1:])


def _parse_tree(at, tree_dump):
    tree = at.add_tree()
    stack = [(tree.root(), json.loads(tree_dump))]

    while len(stack) > 0:
        node, node_json = stack.pop()
        if "leaf" not in node_json:
            children = {child["nodeid"]                        : child for child in node_json["children"]}

            feat_id = feat2id_map(node_json["split"])
            if "split_condition" in node_json:
                split_value = float(node_json["split_condition"])
                tree.split(node, feat_id, split_value)
                left_id = node_json["yes"]
                right_id = node_json["no"]
            else:
                tree.split(node, feat_id)  # binary split
                # (!) this is reversed -> LtSplit(_, 1.0) -> 0.0 goes left
                left_id = node_json["no"]
                right_id = node_json["yes"]

            stack.append((tree.right(node), children[right_id]))
            stack.append((tree.left(node), children[left_id]))

        else:
            leaf_value = node_json["leaf"]
            tree.set_leaf_value(node, 0, leaf_value)


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
    num_leaf_values = ensemble.n_classes_

    if "Regressor" in type(ensemble).__name__:
        print("SKLEARN: regressor")

        def extract_value_fun(v, i):
            # print("skl leaf regr", v)
            return v[0]/num_trees
    elif "Classifier" in type(ensemble).__name__:
        print(f"SKLEARN: classifier with {num_leaf_values} classes")

        def extract_value_fun(v, i):
            # print("skl leaf clf", v[0], sum(v[0]), v[0][i])
            return v[0][i]/sum(v[0])/num_trees
    else:
        raise RuntimeError("cannot determine extract_value_fun for:",
                           type(ensemble).__name__)

    at = ConAddTree(num_leaf_values)
    for tree in ensemble.estimators_:
        addtree_sklearn_tree(at, tree.tree_, extract_value_fun)
    return at


def addtree_sklearn_ensemble_mlticlass(ensemble):
    addtrees = []
    num_trees = len(ensemble.estimators_)
    num_classes = ensemble.n_classes_
    for i in range(num_classes):
        def extract_value_fun(v): return (v[0][i]/sum(v[0]))/num_trees
        at = addtree_sklearn_ensemble(ensemble, extract_value_fun)
        addtrees.append(at)
    return addtrees
