import codecs
from io import StringIO

from .pytreeck import *

def __realdomain__str(self):
    return "[{:.3g}, {:.3g})".format(self.lo, self.hi)

def __realdomain__eq(self, o):
    return self.lo == o.lo and self.hi == o.hi

def __realdomain__hash(self):
    return hash((self.lo, self.hi))

def __booldomain__str(self):
    return "{True}" if self.is_true() \
        else "{False}" if self.is_false() \
        else "{False, True}"

def __booldomain__eq(self, o):
    return self._value == o._value

def __booldomain__hash(self):
    return hash(self._value)

RealDomain.__str__ = __realdomain__str
RealDomain.__eq__ = __realdomain__eq
RealDomain.__hash__ = __realdomain__hash
BoolDomain.__str__ = __booldomain__str
BoolDomain.__eq__ = __booldomain__eq
BoolDomain.__hash__ = __booldomain__hash

def __tree_predict_single(self, example):
    node = self.root()
    while not self.is_leaf(node):
        split = self.get_split(node) # ("lt", feat_id, split_value) OR ("bool", feat_id)
        value = example[split[1]]
        if split[0] == "lt":
            assert isinstance(value, float), f"is {type(value)} instead"
            go_left = value < split[2]
        elif split[0] == "bool": # false left, true right
            assert isinstance(value, bool), f"is {type(value)} instead"
            go_left = not value
        node = self.left(node) if go_left else self.right(node)
    return self.get_leaf_value(node)

def __tree_predict(self, examples):
    return list(map(self.predict_single, examples))

Tree.predict_single = __tree_predict_single
Tree.predict = __tree_predict

def __addtree_iter(self):
    for i in range(len(self)):
        yield self[i]

def __addtree_predict_single(self, example):
    result = self.base_score
    for tree in self:
        result += tree.predict_single(example)
    return result

def __addtree_predict(self, examples):
    predictions = []
    #print("predicting...", end="")
    for i, example in enumerate(examples):
        #print("\rpredicting...", i, "/", len(examples), end="")
        predictions.append(self.predict_single(example))
    #print("\rdone                    ")
    return predictions

def __addtree_write(self, f):
    with open(f, "w") as fh:
        json = self.to_json()
        fh.write(json)

def __addtree_read(f):
    with open(f, "r") as fh:
        json = fh.read()
        return AddTree.from_json(json)

AddTree.__iter__ = __addtree_iter
AddTree.predict_single = __addtree_predict_single
AddTree.predict = __addtree_predict
AddTree.write = __addtree_write
AddTree.read = __addtree_read


class AddTreeFeatureTypes:
    def __init__(self, at):
        self._types = dict()

        for tree_index in range(len(at)):
            tree = at[tree_index]
            self._check_types(tree, tree.root())

    def _check_types(self, tree, node):
        l, r = tree.left(node), tree.right(node)

        if not tree.is_internal(node): return

        split = tree.get_split(node)
        split_type = split[0]
        feat_id = split[1]

        if feat_id in self._types and self._types[feat_id] != split_type:
            raise RuntimeError(f"AddTree split type error for feat_id {feat_id}")

        self._types[feat_id] = split_type

        if not tree.is_leaf(l): self._check_types(tree, l)
        if not tree.is_leaf(r): self._check_types(tree, r)

    def feat_ids(self):
        yield from self._types.keys()

    def feature_types(self):
        yield from self._types.items()

    def __getitem__(self, feat_id):
        if feat_id not in self._types:
            raise KeyError(f"unknown feat_id {feat_id}")
        return self._types[feat_id]
