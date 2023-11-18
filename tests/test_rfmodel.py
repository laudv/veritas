import numpy as np

from pprint import pprint

from veritas import AddTree


def round_splits(t0, i0, t1, i1):
    if t0.is_internal(i0):
        split = t0.get_split(i0)

        splitval = np.floor(split.split_value) + 0.5

        t1.split(i1, split.feat_id, splitval)

        round_splits(t0, t0.left(i0), t1, t1.left(i1))
        round_splits(t0, t0.right(i0), t1, t1.right(i1))
    else:
        for i in range(t0.num_leaf_values()):
            v = t0.get_leaf_value(i0, i)
            t1.set_leaf_value(i1, i, v)

def transform_addtree(at0):
    at1 = AddTree(at0.num_leaf_values())
    for t0 in at0:
        t1 = at1.add_tree()
        round_splits(t0, t0.root(), t1, t1.root())
    for i in range(at0.num_leaf_values()):
        v = at0.get_base_score(i)
        at1.set_base_score(i, v)
    return at1


if __name__ == "__main__":
    at0 = AddTree.read("tests/models/rf_model.json")
    at1 = transform_addtree(at0)

    pprint(at0.get_splits())
    print()
    pprint(at1.get_splits())
