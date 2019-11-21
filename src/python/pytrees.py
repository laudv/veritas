import z3

class PyTrees:
    
    def __init__(self, offsets, feat_ids, values, lefts):
        self._offsets = offsets
        self._lefts = lefts
        self._feat_ids = feat_ids
        self._values = values

    def trees(self):
        yield from range(len(self._offsets))

    def index(self, tree, node):
        return self._offsets[tree] + node

    def is_root(self, tree, node):
        return self._feat_ids[self.index(tree, node)] == 0

    def is_leaf(self, tree, node):
        return self._feat_ids[self.index(tree, node)] == -1

    def is_internal(self, tree, node):
        return not self.is_leaf(tree, node)

    def split_value(self, tree, node):
        assert self.is_internal(tree, node)
        return self._values[self.index(tree, node)]

    def leaf_value(self, tree, node):
        assert self.is_leaf(tree, node)
        return self._values[self.index(tree, node)]

    def left(self, tree, node):
        return self._lefts[self.index(tree, node)]

    def right(self, tree, node):
        return self._lefts[self.index(tree, node)] + 1

    #def _encode_tree(self, tree, wvar, node, modeled_leafs):
    #    if tree.is_leaf(node):
    #        return (wvar == tree.get_leaf_value(node))
    #    elif node in modeled_leafs:
    #        lower, upper = tree.get_leaf_value_bounds(node)
    #        return z3.And((wvar >= lower), (wvar <= upper))
    #    else:
    #        l, r = tree.left(node), tree.right(node)
    #        lc = self._encode_tree(tree, wvar, l, modeled_leafs)
    #        rc = self._encode_tree(tree, wvar, r, modeled_leafs)
    #        c = self._get_split_constraint(tree.get_split(node))
    #        return z3.Or(z3.And(c, lc), z3.And(z3.Not(c), rc))
