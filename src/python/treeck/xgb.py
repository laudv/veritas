import json

from xgboost.sklearn import XGBModel
from xgboost.core import Booster

from . import AddTree

def addtree_from_xgb_model(model, feat2id_map=lambda x: int(x[1:])):
    base_score = 0.5
    if isinstance(model, XGBModel):
        base_score = model.base_score
        model = model.get_booster()
    assert isinstance(model, Booster)

    dump = model.get_dump("", dump_format="json")
    at = AddTree()
    at.base_score = base_score

    for tree_dump in dump:
        _parse_tree(at, tree_dump, feat2id_map)

    return at
    
def _parse_tree(at, tree_dump, feat2id_map):
    tree = at.add_tree()
    stack = [(tree.root(), json.loads(tree_dump))]

    while len(stack) > 0:
        node, node_json = stack.pop()
        if "leaf" not in node_json:
            feat_id = feat2id_map(node_json["split"])
            split_value = node_json["split_condition"]
            tree.split(node, feat_id, split_value)

            # let's hope the ordering of "children" is [left,right]
            stack.append((tree.right(node), node_json["children"][1]))
            stack.append((tree.left(node), node_json["children"][0]))
        else:
            leaf_value = node_json["leaf"]
            tree.set_leaf_value(node, leaf_value)
