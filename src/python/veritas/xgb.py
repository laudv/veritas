# \file xgb.py
#
# This requires `xgboost` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import json
import numpy as np

# from xgboost.sklearn import XGBModel
# from xgboost.core import Booster as xgbbooster

from . import AddTree, AddTreeType, AddTreeConverter

class XGB_AddTreeConverter(AddTreeConverter):
    def get_addtree(self,model):
        try:
            model = model.get_booster()
        except:
            pass
        
        # if isinstance(model, XGBModel):
        #     model = model.get_booster()
        # assert isinstance(
        #     model, xgbbooster), f"not xgb.Booster but {type(model)}"

        param_dump = json.loads(model.save_config())['learner']
        print(param_dump)
        base_score = float(param_dump['learner_model_param']["base_score"])
        model_type = param_dump["objective"]["name"]
        if "multi" in model_type:
            num_class = int(param_dump['learner_model_param']["num_class"])
            return multi_addtree_xgb(model,num_class,base_score)
        elif "logistic" in model_type:
            base_score = 0.5
            # -------------- This is still not fixed it seems --------------
            # Base_score is set to 0.5 but produces an offset of 0.5
            # Base_margin is porbably used but unable to retrieve from xgboost
            # https://xgboost.readthedocs.io/en/stable/prediction.html#base-margin
            return addtree_xgb(model, base_score, type_=AddTreeType.GB_CLF)
        return addtree_xgb(model, base_score, type_=AddTreeType.GB_REGR)


def multi_addtree_xgb(model, num_class, base_score):
    ats = [addtree_xgb(model, base_score, type_=AddTreeType.GB_MULTI, multiclass=(clazz, num_class)) for clazz in range(num_class)]
    at = ats[0].make_multiclass(0, num_class)
    for k in range(1, num_class):
        at.add_trees(ats[k], k)
    return at


def addtree_xgb(model, base_score, type_=AddTreeType.RAW, multiclass=(0, 1)):
    dump = model.get_dump("", dump_format="json")

    at = AddTree(1, type_)
    offset, num_classes = multiclass
    at.set_base_score(0, base_score)

    for i in range(offset, len(dump), num_classes):
        _parse_tree_xgb(at, dump[i])

    return at


def xgb_feat2id_map(f): return int(f[1:])


def _parse_tree_xgb(at, tree_dump):
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
