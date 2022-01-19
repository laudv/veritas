## \file groot.py
# Convert GROOT trees to Veritas AddTree instances.
# GROOT: https://github.com/tudelft-cda-lab/GROOT
#
# Copyright 2020 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import numpy as np
from . import AddTree
from .sklearn import RfAddTree

import groot.model

def _addtree_from_groot_tree(at, gtree, extract_value_fun):
    vtree = at.add_tree()
    stack = [(vtree.root(), gtree.root_)]

    while len(stack) > 0:
        vnode, gnode = stack.pop()
        if gnode.is_leaf():
            leaf_value = extract_value_fun(gnode.value)
            vtree.set_leaf_value(vnode, leaf_value)
        else:
            feat_id = gnode.feature
            thrs = gnode.threshold
            split_value = np.nextafter(np.float32(thrs), np.float32(np.inf)) # <= splits
            vtree.split(vnode, feat_id, split_value)
            stack.append((vtree.right(vnode), gnode.right_child))
            stack.append((vtree.left(vnode), gnode.left_child))

def addtree_from_groot_ensemble(model, extract_value_fun=None):
    assert isinstance(model, groot.model.GrootRandomForestClassifier), f"not GrootRandomForestClassifier but {type(model)}"

    num_trees = len(model.estimators_)
    if extract_value_fun is None:
        if "Classifier" in type(model).__name__:
            print("GROOT Classifier")
            extract_value_fun = lambda v: v[1]
        else:
            raise RuntimeError(f"{type(model).__name__} not supported")

    at = RfAddTree()
    for tree in model.estimators_:
        _addtree_from_groot_tree(at, tree, extract_value_fun)
    at.base_score = -num_trees / 2
    return at

