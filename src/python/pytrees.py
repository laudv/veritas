class PyTrees:
    
    def __init__(self, offsets, lefts, feat_ids, values):
        self._offsets = offsets
        self._lefts = lefts
        self._feat_ids = feat_ids
        self._values = values

    def __len__(self):
        return len(self._offsets)

    def num_nodes(self):
        return len(self._lefts)

    def trees(self):
        yield from range(len(self))

    def index(self, tree, node):
        return self._offsets[tree] + node

    def root(self, tree):
        return 0

    def is_root(self, tree, node):
        return self._feat_ids[self.index(tree, node)] == 0

    def is_leaf(self, tree, node):
        return self._feat_ids[self.index(tree, node)] == -1

    def is_internal(self, tree, node):
        return not self.is_leaf(tree, node)

    def split_value(self, tree, node):
        assert self.is_internal(tree, node)
        return self._values[self.index(tree, node)]

    def split_feat_id(self, tree, node):
        assert self.is_internal(tree, node)
        return self._feat_ids[self.index(tree, node)]

    def leaf_value(self, tree, node):
        assert self.is_leaf(tree, node)
        return self._values[self.index(tree, node)]

    def left(self, tree, node):
        return self._lefts[self.index(tree, node)]

    def right(self, tree, node):
        return self._lefts[self.index(tree, node)] + 1
