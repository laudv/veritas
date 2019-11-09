from .treeck import *

def __tree_predict_single(zelf, example):
    node = zelf.root()
    while not node.is_leaf():
        split = node.get_split()
        go_left = split.test(example[split.feat_id])
        node = node.left() if go_left else node.right()
    return node.leaf_value()

def __tree_predict(zelf, examples):
    return list(map(zelf.predict_single, examples))

def __tree_write(zelf, f):
    with open(f, "w") as fh:
        json = zelf.to_json()
        fh.write(json)

def __tree_read(f):
    with open(f, "r") as fh:
        json = fh.read()
        return Tree.from_json(json)

Tree.predict_single = __tree_predict_single
Tree.predict = __tree_predict
Tree.write = __tree_write
Tree.read = __tree_read
