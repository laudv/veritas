# \file lgb.py
#
# This requires `lightgbm` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

from . import AddTree, AddTreeType, AddTreeConverter

# from lightgbm import LGBMModel
# from lightgbm import Booster as lgbmbooster

import numpy as np


class LGB_AddTreeConverter(AddTreeConverter):
    def get_addtree(self,model):
        try:
            model = model.booster_
        except:
            pass
        
        # if isinstance(model, LGBMModel):
        #     model = model.booster_
        # assert isinstance(
        #     model, lgbmbooster), f"not xgb.Booster but {type(model)}"

        dump = model.dump_model()
        num_class = dump["num_class"]
        type_ = dump["objective"]
        if num_class > 2:
            return multi_addtree_lgbm(model,num_class)
        if "binary" in type_:
            return addtree_lgbm(model, type_=AddTreeType.GB_CLF)
        else:
            return addtree_lgbm(model, type_=AddTreeType.GB_REGR)


def multi_addtree_lgbm(model,num_class):
    ats = [addtree_lgbm(model, type_=AddTreeType.GB_MULTI, multiclass=(clazz, num_class)) for clazz in range(num_class)]
    at = ats[0].make_multiclass(0, num_class)
    for k in range(1, num_class):
        at.add_trees(ats[k], k)
    return at


def addtree_lgbm(model, type_=AddTreeType.RAW, multiclass=(0, 1)):
    dump = model.dump_model()
    at = AddTree(1, type_)

    offset, num_classes = multiclass

    trees = dump["tree_info"]
    for i in range(offset, len(trees), num_classes):
        _parse_tree_lgbm(at, trees[i]["tree_structure"])

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
