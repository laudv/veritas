import json
import numpy as np
import math

from xgboost.sklearn import XGBModel
from xgboost.core import Booster as xgbbooster

import sklearn.tree as sktree
from sklearn.ensemble import _forest

from lightgbm import LGBMModel
from lightgbm import Booster as lgbmbooster

from . import AddTree, AddTreeType


def get_addtree(model):
    module_name = getattr(model, '__module__', None)

    # XGB
    if "xgboost" in str(module_name):
        if isinstance(model, XGBModel):
            model = model.get_booster()
        assert isinstance(
            model, xgbbooster), f"not xgb.Booster but {type(model)}"
        param_dump = json.loads(model.save_config())['learner']
        base_score = float(param_dump['learner_model_param']["base_score"])
        model_type = param_dump["objective"]["name"]
        if "multi" in model_type:
            num_class = int(param_dump['learner_model_param']["num_class"])
            return [addtree_xgb(model, base_score, type_=AddTreeType.GB_MULTI, multiclass=(clazz, num_class)) for clazz in range(num_class)]
        elif "logistic" in model_type:
            base_score = 0.0
            # Base_score is set to 0.5 but produces an offset of 0.5
            # Base_margin is porbably used but unable to retrieve from xgboost
            # https://xgboost.readthedocs.io/en/stable/prediction.html#base-margin
            return addtree_xgb(model, base_score, type_=AddTreeType.GB_CLF)
        return addtree_xgb(model, base_score, type_=AddTreeType.GB_REGR)

    # Sklearn RandomForest / InsulationForest in the Future?
    elif "sklearn.ensemble._forest" in str(module_name):
        return addtree_sklearn_ensemble(model)

    # LGBM
    elif "lightgbm" in str(module_name):
        if isinstance(model, LGBMModel):
            model = model.booster_
        assert isinstance(
            model, lgbmbooster), f"not xgb.Booster but {type(model)}"
        dump = model.dump_model()
        num_class = dump["num_class"]
        type_ = dump["objective"]
        if num_class > 2:
            return [addtree_lgbm(model, type_=AddTreeType.GB_MULTI, multiclass=(clazz, num_class)) for clazz in range(num_class)]
        if "binary" in type_:
            return addtree_lgbm(model, type_=AddTreeType.GB_CLF)
        else:
            return addtree_lgbm(model, type_=AddTreeType.GB_REGR)

    return -1


def addtree_xgb(model, base_score, type_=AddTreeType.RAW, multiclass=(0, 1)):
    dump = model.get_dump("", dump_format="json")

    at = AddTree(1, type_)
    offset, num_classes = multiclass
    at.set_base_score(0, base_score)

    for i in range(offset, len(dump), num_classes):
        _parse_tree_xgb(at, dump[i])

    return at


def addtree_lgbm(model, type_=AddTreeType.RAW, multiclass=(0, 1)):
    dump = model.dump_model()
    at = AddTree(1, type_)

    offset, num_classes = multiclass

    trees = dump["tree_info"]
    for i in range(offset, len(trees), num_classes):
        _parse_tree_lgbm(at, trees[i]["tree_structure"])

    return at


def xgb_feat2id_map(f): return int(f[1:])


def _parse_tree_xgb(at, tree_dump):
    tree = at.add_tree()
    stack = [(tree.root(), json.loads(tree_dump))]

    while len(stack) > 0:
        node, node_json = stack.pop()
        if "leaf" not in node_json:
            children = {child["nodeid"]: child for child in node_json["children"]}

            feat_id = xgb_feat2id_map(node_json["split"])

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
    num_leaf_values = 1

    if "Regressor" in type(ensemble).__name__:
        print("SKLEARN: regressor")
        type_ = AddTreeType.RF_REGR

        def extract_value_fun(v, i):
            # print("skl leaf regr", v)
            return v[0]
    elif "Classifier" in type(ensemble).__name__:
        num_leaf_values = ensemble.n_classes_  # TODO: Change to 1 if num_class < 2
        type_ = AddTreeType.RF_CLF
        print(f"SKLEARN: classifier with {num_leaf_values} classes")

        def extract_value_fun(v, i):
            # print("skl leaf clf", v[0], sum(v[0]), v[0][i])
            # TODO: Remove mean --> binding/addtree.py predict()/predict_proba()
            return v[0][i]/sum(v[0])/num_trees
    else:
        raise RuntimeError("cannot determine extract_value_fun for:",
                           type(ensemble).__name__)

    at = AddTree(num_leaf_values, type_)
    for tree in ensemble.estimators_:
        addtree_sklearn_tree(at, tree.tree_, extract_value_fun)
    return at


def _parse_tree_lgbm(at, tree_json):
    tree = at.add_tree()
    stack = [(tree.root(), tree_json)]

    while len(stack) > 0:
        node, node_json = stack.pop()
        try:
            if "split_feature" in node_json:
                feat_id = node_json["split_feature"]
                if not node_json["default_left"]:
                    print("warning: default_left != True not supported")
                if node_json["decision_type"] == "<=":
                    split_value = np.float32(node_json["threshold"])
                    split_value = np.nextafter(
                        split_value, -np.inf, dtype=np.float32)
                    tree.split(node, feat_id, split_value)
                    left = node_json["left_child"]
                    right = node_json["right_child"]
                else:
                    raise RuntimeError(
                        f"not supported decision_type {node_json['decision_type']}")

                stack.append((tree.right(node), right))
                stack.append((tree.left(node), left))

            else:
                leaf_value = node_json["leaf_value"]
                tree.set_leaf_value(node, 0, leaf_value)
        except KeyError as e:
            print("error", node_json.keys())
            raise e
