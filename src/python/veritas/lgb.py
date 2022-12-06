## \file lgb.py
#
# This requires `lightgbm` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

from lightgbm import Booster
import numpy as np

from .xgb import GbAddTree

def addtrees_from_multiclass_lgb_model(model, nclasses, feat2id_map=int):
    pass

def addtree_from_lgb_model(model, feat2id_map=int):
    """
    mulclass=(offset, num_classes): only loads tree offset, offset+num_classes,
    offset+2*num_classes...
    """
    base_score = 0.0
    # TODO lightgm scikitlearn interface
    #if isinstance(model, XGBModel):
    #    base_score = model.base_score if model.base_score is not None else base_score
    #    model = model.get_booster()
    assert isinstance(model, Booster), f"not lgb.Booster but {type(model)}"

    dump = model.dump_model()

    at = GbAddTree()
    at.base_score = base_score

    for tree_json in dump["tree_info"]:
        _parse_tree(at, tree_json["tree_structure"], feat2id_map)

    return at

def _parse_tree(at, tree_json, feat2id_map):
    tree = at.add_tree()
    stack = [(tree.root(), tree_json)]

    while len(stack) > 0:
        node, node_json = stack.pop()
        try:
            if "split_feature" in node_json:
                feat_id = feat2id_map(node_json["split_feature"])
                if not node_json["default_left"]:
                    print("warning: default_left != True not supported")
                if node_json["decision_type"] == "<=":
                    split_value = np.float32(node_json["threshold"])
                    split_value = np.nextafter(split_value, -np.inf, dtype=np.float32)
                    tree.split(node, feat_id, split_value)
                    left = node_json["left_child"]
                    right = node_json["right_child"]
                else:
                    raise RuntimeError(f"not supported decision_type {node_json['decision_type']}")

                stack.append((tree.right(node), right))
                stack.append((tree.left(node), left))

            else:
                leaf_value = node_json["leaf_value"]
                tree.set_leaf_value(node, leaf_value)
        except KeyError as e:
            print("error", node_json.keys())
            raise e
