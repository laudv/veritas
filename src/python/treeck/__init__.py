# Copyright 2019 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

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

def __tree_predict_leaf(self, example):
    node = self.root()
    while not self.is_leaf(node):
        split = self.get_split(node)
        value = example[split.feat_id]
        if isinstance(split, LtSplit):
            #assert isinstance(value, float), f"is {type(value)} instead"
            go_left = split.test(value)
        elif isinstance(split, BoolSplit):
            #assert isinstance(value, bool), f"is {type(value)} instead"
            go_left = split.test(value)
        node = self.left(node) if go_left else self.right(node)
    return node

def __tree_predict_single(self, example):
    node = __tree_predict_leaf(self, example)
    return self.get_leaf_value(node)

def __tree_predict(self, examples):
    return list(map(self.predict_single, examples))

Tree.predict_leaf = __tree_predict_leaf
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
    it = enumerate(examples)
    try:
        import pandas as pd
        if isinstance(examples, pd.DataFrame):
            it = examples.iterrows()
    except: pass

    predictions = []
    #print("predicting...", end="")
    for i, example in it:
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

        # sort by key
        self._types = {fid: self._types[fid] for fid in sorted(self._types)}

    def _check_types(self, tree, node):
        l, r = tree.left(node), tree.right(node)

        if not tree.is_internal(node): return

        split = tree.get_split(node)
        split_type = type(split)

        if split.feat_id in self._types and self._types[split.feat_id] != split_type:
            raise RuntimeError(f"AddTree split type error for feat_id {split.feat_id}")

        self._types[split.feat_id] = split_type

        if not tree.is_leaf(l): self._check_types(tree, l)
        if not tree.is_leaf(r): self._check_types(tree, r)

    def feat_ids(self):
        yield from self._types.keys()

    def __iter__(self):
        yield from self._types.items()

    def __getitem__(self, feat_id):
        if feat_id not in self._types:
            raise KeyError(f"unknown feat_id {feat_id}")
        return self._types[feat_id]


def get_closest_instance(base_instance, doms):
    instance = base_instance.copy()
    for key, dom in doms.items():
        assert isinstance(dom, RealDomain)
        v = instance[key]
        if dom.contains(v):
            continue # keep the value

        dist_lo = abs(dom.lo - v)
        dist_hi = abs(v - dom.hi)
        if dist_lo < dist_hi:
            instance[key] = dom.hi - ((dom.hi-dom.lo) / 1000) # hi is not included
        else:
            instance[key] = dom.lo
    return instance
