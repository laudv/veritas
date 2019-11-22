from io import StringIO

from .pytreeck import *

from .consts import LESS_THAN, GREATER_THAN
from .pytrees import PyTrees
from .z3solver import Z3Solver
from .parallel_solver import ParallelSolver

def __realdomain__str(self):
    return "[{:.3g}, {:.3g})".format(self.lo, self.hi)

RealDomain.__str__ = __realdomain__str

def __ltsplit__str(self):
    return "X{} < {:.3g}".format(self.feat_id, self.split_value)

def __ltsplit__eq(self, other):
    return self.feat_id == other.feat_id \
            and self.split_value == other.split_value

LtSplit.__str__ = __ltsplit__str
LtSplit.__eq__ = __ltsplit__eq

def __node__eq(self, other):
    return self.is_internal() == other.is_internal() \
            and (not self.is_internal() or (self.get_split() == other.get_split())) \
            and (not self.is_leaf() or (self.leaf_value() == other.leaf_value()))

Node.__eq__ = __node__eq

def __tree_predict_single(self, example):
    node = self.root()
    while not node.is_leaf():
        split = node.get_split()
        go_left = split.test(example[split.feat_id])
        node = node.left() if go_left else node.right()
    return node.leaf_value()

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
    return list(map(self.predict_single, examples))

def __addtree_write(self, f):
    with open(f, "w") as fh:
        json = self.to_json()
        fh.write(json)

def __addtree_read(f):
    with open(f, "r") as fh:
        json = fh.read()
        return AddTree.from_json(json)

def __addtree_into_pytrees(self):
    return PyTrees(*self._export_lists())

AddTree.__iter__ = __addtree_iter
AddTree.predict_single = __addtree_predict_single
AddTree.predict = __addtree_predict
AddTree.write = __addtree_write
AddTree.read = __addtree_read
AddTree.pytrees = __addtree_into_pytrees

def __searchspace_into_pytrees(self, leaf_id):
    return PyTrees(*self._export_lists(leaf_id))

SearchSpace.get_pruned_pytrees = __searchspace_into_pytrees;
