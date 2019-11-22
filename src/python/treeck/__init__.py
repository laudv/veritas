from io import StringIO

from .pytreeck import *

from .consts import LESS_THAN, GREATER_THAN
from .z3solver import Z3Solver
from .parallel_solver import ParallelSolver

def __realdomain__str(self):
    return "[{:.3g}, {:.3g})".format(self.lo, self.hi)

RealDomain.__str__ = __realdomain__str

#def __ltsplit__str(self):
#    return "X{} < {:.3g}".format(self.feat_id, self.split_value)
#
#def __ltsplit__eq(self, other):
#    return self.feat_id == other.feat_id \
#            and self.split_value == other.split_value
#
#LtSplit.__str__ = __ltsplit__str
#LtSplit.__eq__ = __ltsplit__eq

#def __node__eq(self, other):
#    return self.is_internal() == other.is_internal() \
#            and (not self.is_internal() or (self.get_split() == other.get_split())) \
#            and (not self.is_leaf() or (self.leaf_value() == other.leaf_value()))
#
#Node.__eq__ = __node__eq

def __tree_predict_single(self, example):
    node = self.root()
    while not self.is_leaf(node):
        fid, sv = self.get_split(node)
        go_left = example[fid] < sv
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
    print("predicting...", end="")
    for i, example in enumerate(examples):
        print("\rpredicting...", i, "/", len(examples), end="")
        predictions.append(self.predict_single(example))
    print("\rdone                    ")
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
