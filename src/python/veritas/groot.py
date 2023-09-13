## \file groot.py
# Convert GROOT trees to Veritas AddTree instances.
# GROOT: https://github.com/tudelft-cda-lab/GROOT
#
# This requires `groot` to be installed.
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import numpy as np
from . import AddTree, Interval

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
            box = vtree.compute_box(vnode)
            doml, domr = Interval().split(split_value)
            if feat_id in box:
                validl, validr = doml.overlaps(box[feat_id]), domr.overlaps(box[feat_id])
                if not validl or not validr:
                    print(f"WARNING: invalid split, node interval of feat {feat_id} is {box[feat_id]}",
                          f"but split value is {split_value} (node {vnode})")
                    #print(vtree, end="")
                if not validl:
                    print(" -> considering only right branch, not adding split")
                    stack.append((vnode, gnode.right_child))
                    continue
                elif not validr:
                    print(" -> considering only left branch, not adding split")
                    stack.append((vnode, gnode.left_child))
                    continue
            vtree.split(vnode, feat_id, split_value)
            stack.append((vtree.right(vnode), gnode.right_child))
            stack.append((vtree.left(vnode), gnode.left_child))

def addtree_from_groot_ensemble(model, extract_value_fun=None):
    assert isinstance(model, groot.model.GrootRandomForestClassifier),\
        f"not GrootRandomForestClassifier but {type(model)}"

    num_trees = len(model.estimators_)
    if extract_value_fun is None:
        if "Classifier" in type(model).__name__:
            print("GROOT Classifier")
            extract_value_fun = lambda v: v[1]
        else:
            raise RuntimeError(f"{type(model).__name__} not supported")

    at = AddTree(1)
    for i, gtree in enumerate(model.estimators_):
        _addtree_from_groot_tree(at, gtree, extract_value_fun)
        #print(at[len(at)-1])
        #print(gtree.root_.pretty_print())
    at.base_score = -num_trees / 2
    return at

