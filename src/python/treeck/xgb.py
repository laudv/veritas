# Copyright 2019 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import json

from xgboost.sklearn import XGBModel
from xgboost.core import Booster

from . import AddTree

def addtree_from_xgb_model(model, feat2id_map=lambda x: int(x[1:]),
        multiclass=(0, 1)):
    """
    mulclass=(offset, num_classes): only loads tree offset, offset+num_classes,
    offset+2*num_classes...
    """
    base_score = 0.5
    if isinstance(model, XGBModel):
        base_score = model.base_score
        model = model.get_booster()
    assert isinstance(model, Booster), f"not xgb.Booster but {type(model)}"

    dump = model.get_dump("", dump_format="json")
    at = AddTree()
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
            feat_id = feat2id_map(node_json["split"])
            if "split_condition" in node_json:
                split_value = node_json["split_condition"]
                tree.split(node, feat_id, split_value)
            else:
                tree.split(node, feat_id) # binary split

            # let's hope the ordering of "children" is [left,right]
            left_id = node_json["yes"]
            right_id = node_json["no"]
            if "missing" in node_json:
                assert node_json["missing"] == left_id, "XGB sparse not supported, set missing=None"

            children = { child["nodeid"]: child for child in node_json["children"] }

            stack.append((tree.right(node), children[right_id]))
            stack.append((tree.left(node), children[left_id]))
        else:
            leaf_value = node_json["leaf"]
            tree.set_leaf_value(node, leaf_value)
