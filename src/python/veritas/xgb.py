## \file xgb.py
#
# This requires `xgboost` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import json
import numpy as np

from xgboost.sklearn import XGBModel
from xgboost.core import Booster

from . import AddTree

class GbAddTree(AddTree):
    def predict_proba(self, X):
        return 1/(1+np.exp(-self.eval(X)))

def addtrees_from_multiclass_xgb_model(model, nclasses, feat2id_map=int):
    return [
        addtree_from_xgb_model(model, feat2id_map, multiclass=(clazz, nclasses))
        for clazz in range(nclasses)
    ]

def addtree_from_xgb_model(model, feat2id_map=int,
        multiclass=(0, 1)):
    """
    mulclass=(offset, num_classes): only loads tree offset, offset+num_classes,
    offset+2*num_classes...
    """
    #base_score = 0.5
    base_score = 0.0 # xgboost 1.4.2
    if isinstance(model, XGBModel):
        base_score = model.base_score if model.base_score is not None else base_score
        model = model.get_booster()
    assert isinstance(model, Booster), f"not xgb.Booster but {type(model)}"

    dump = model.get_dump("", dump_format="json")
    at = GbAddTree()

    at.base_score = base_score
    offset, num_classes = multiclass

    for i in range(offset, len(dump), num_classes):
        _parse_tree(at, dump[i], feat2id_map)

    return at
    
def _parse_tree(at, tree_dump, feat2id_map):
    tree = at.add_tree()
    stack = [(tree.root(), json.loads(tree_dump))]

    while len(stack) > 0:
        node, node_json = stack.pop()
        if "leaf" not in node_json:
            children = { child["nodeid"]: child for child in node_json["children"] }

            feat_id = feat2id_map(node_json["split"])
            if "split_condition" in node_json:
                split_value = float(node_json["split_condition"])
                tree.split(node, feat_id, split_value)
                left_id = node_json["yes"]
                right_id = node_json["no"]
            else:
                tree.split(node, feat_id) # binary split
                left_id = node_json["no"] # (!) this is reversed -> LtSplit(_, 1.0) -> 0.0 goes left
                right_id = node_json["yes"]

            stack.append((tree.right(node), children[right_id]))
            stack.append((tree.left(node), children[left_id]))

        else:
            leaf_value = node_json["leaf"]
            tree.set_leaf_value(node, leaf_value)
