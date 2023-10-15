# \file xgb.py
#
# This requires `xgboost` to be installed.
#
# Copyright 2023 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos, Alexander Schoeters

import json
import numpy as np

from . import AddTree, AddTreeType, AddTreeConverter
from . import InapplicableAddTreeConverter
from . import FloatT

class XGBAddTreeConverter(AddTreeConverter):
    def convert(self,model):
        try:
            from xgboost import XGBModel
            from xgboost.core import Booster as XGBBooster
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("xgb not installed")
        
        if isinstance(model, XGBModel):
            model = model.get_booster()

        if not isinstance(model, XGBBooster):
            raise InapplicableAddTreeConverter("not an xgb model")

        param_dump = json.loads(model.save_config())['learner']
        objective = param_dump["objective"]["name"]

        at = None

        if "multi" in objective:
            num_class = int(param_dump['learner_model_param']["num_class"])
            at = multi_addtree_xgb(model, num_class)
        elif "logistic" in objective:
            at = addtree_xgb(model, at_type=AddTreeType.GB_BINARY)
        else:
            at = addtree_xgb(model, at_type=AddTreeType.GB_REGR)

        base_score_dump = float(param_dump['learner_model_param']["base_score"])
        base_score = try_determine_base_score(model, at)

        if np.max(np.abs(base_score_dump - base_score)) > 1e-5:
            print(f"Warning! XGBoost's repored base_score of {base_score_dump:.6f}",
                  f"does not match manually determined score of {base_score:.6f}")
        else:
            print("XGB converter: manually determined base_score",
                  base_score, f" (dump value {base_score_dump:.6f})")

        if at.num_leaf_values() > 1:
            for k in range(at.num_leaf_values()):
                at.set_base_score(k, base_score[k])
        else:
            at.set_base_score(0, base_score)

        return at

def try_determine_base_score(booster, at, seed=472934901, n=1000):
    import xgboost as xgb
    num_features = booster.num_features()

    rng = np.random.default_rng(seed)
    x = rng.random((n, num_features), dtype=FloatT)

    for k, vs in at.get_splits().items():
        vmin, vmax = min(vs), max(vs)
        vdiff = vmax-vmin
        vmin -= 0.05 * vdiff
        vmax += 0.05 * vdiff
        x[:, k] = x[:, k] * (vmax-vmin) + vmin

    pred0 = booster.predict(xgb.DMatrix(x), output_margin=True)
    pred1 = at.eval(x)

    if pred1.shape[1] == 1:
        pred1 = pred1.reshape(pred1.shape[0])

    pred0 = pred0.astype(FloatT).mean(axis=0)
    pred1 = pred1.astype(FloatT).mean(axis=0)

    return pred0 - pred1

def multi_addtree_xgb(model, num_class):
    at0 = addtree_xgb(model, at_type=AddTreeType.GB_MULTI,
                      multiclass=(0, num_class))
    at = at0.make_multiclass(0, num_class)
    for k in range(1, num_class):
        atk = addtree_xgb(model, at_type=AddTreeType.GB_MULTI,
                          multiclass=(k, num_class))
        at.add_trees(atk, k)
    return at

def addtree_xgb(model, at_type, multiclass=(0, 1)):
    dump = model.get_dump("", dump_format="json")

    at = AddTree(1, at_type)
    offset, num_classes = multiclass

    for i in range(offset, len(dump), num_classes):
        parse_tree_xgb(at, dump[i])

    return at

def xgb_feat2id_map(f): return int(f[1:])

def parse_tree_xgb(at, tree_dump):
    tree = at.add_tree()
    stack = [(tree.root(), json.loads(tree_dump))]

    while len(stack) > 0:
        node, node_json = stack.pop()
        if "leaf" not in node_json:
            children = {child["nodeid"]
                : child for child in node_json["children"]}

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

    return at
