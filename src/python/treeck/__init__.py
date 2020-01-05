import codecs
from io import StringIO

from .pytreeck import *

def __realdomain__str(self):
    return "[{:.3g}, {:.3g})".format(self.lo, self.hi)

def __realdomain__eq(self, o):
    return self.lo == o.lo and self.hi == o.hi

def __realdomain__hash(self):
    return hash((self.lo, self.hi))

RealDomain.__str__ = __realdomain__str
RealDomain.__eq__ = __realdomain__eq

def __tree_predict_single(self, example):
    node = self.root()
    while not self.is_leaf(node):
        split = self.get_split(node) # ("lt", feat_id, split_value) OR ("bool", feat_id)
        value = example[split[1]]
        if split[0] == "lt":
            assert isinstance(value, float)
            go_left = value < split[2]
        elif split[0] == "bool": # false left, true right
            assert isinstance(value, bool)
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
