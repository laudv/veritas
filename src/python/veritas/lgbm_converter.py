# \file lgbm.py
#
# This requires `lightgbm` to be installed.
#
# Copyright 2023 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos, Alexander Schoeters

from . import AddTree, AddTreeType, AddTreeConverter
from . import InapplicableAddTreeConverter
import numpy as np

class LGBMAddTreeConverter(AddTreeConverter):
    def convert(self,model):
        try:
            from lightgbm import LGBMModel
            from lightgbm import Booster as LGBMBooster
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("lgbm not installed")

        if isinstance(model, LGBMModel):
            model = model.booster_

        if not isinstance(model, LGBMBooster):
            raise InapplicableAddTreeConverter("not a lgbm model")

        dump = model.dump_model()
        num_class = dump["num_class"]
        objective = dump["objective"]
        if num_class > 2:
            return multi_addtree_lgbm(model, num_class)
        if "binary" in objective:
            return addtree_lgbm(model, at_type=AddTreeType.CLF_SOFTMAX)
        else:
            return addtree_lgbm(model, at_type=AddTreeType.REGR)
        

def multi_addtree_lgbm(model, num_class):
    ats = [addtree_lgbm(model,
                        # at_type=AddTreeType.GB_MULTI,
                        at_type=AddTreeType.CLF_SOFTMAX,
                        multiclass=(clazz, num_class))
           for clazz in range(num_class)]

    at = ats[0].make_multiclass(0, num_class)

    for k in range(1, num_class):
        at.add_trees(ats[k], k)

    return at

def addtree_lgbm(model, at_type, multiclass=(0, 1)):
    dump = model.dump_model()
    at = AddTree(1, at_type)

    offset, num_classes = multiclass

    trees = dump["tree_info"]
    for i in range(offset, len(trees), num_classes):
        parse_tree_lgbm(at, trees[i]["tree_structure"])

    return at

def parse_tree_lgbm(at, tree_json):
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
