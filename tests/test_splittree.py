import unittest
import numpy as np

from treeck import *
from treeck.plot import TreePlot

class TestSplitTree(unittest.TestCase):
    def test_single_tree1(self):
        at = AddTree(1)
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        print(at)

        st = SplitTree(at, {})
        st.split_leaf(0)

    def test_single_tree2(self):
        at = AddTree(1)
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 1)
        t.set_leaf_value(t.left(t.root()), 0.1)
        t.split(t.right(t.root()), 0, 2)
        t.set_leaf_value(t.left(t.right(t.root())), 0.2)
        t.split(t.right(t.right(t.root())), 0, 3)
        t.set_leaf_value( t.left(t.right(t.right(t.root()))), 0.3)
        t.set_leaf_value(t.right(t.right(t.right(t.root()))), 0.4)

        print(at)

        st = SplitTree(at, {})
        st.split_leaf(0)

    def test_two_trees1(self):
        at = AddTree(1)
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        t = at.add_tree();
        t.split(t.root(), 0, 3)
        t.split( t.left(t.root()), 1, 1.2)
        t.split(t.right(t.root()), 1, 3.3)
        t.split(t.right(t.right(t.root())), 0, 4)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value( t.left(t.right(t.right(t.root()))), 0.5)
        t.set_leaf_value(t.right(t.right(t.right(t.root()))), 0.6)

        print(at)

        st = SplitTree(at, {})
        st.split_leaf(0)

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-hard.json")
        st = SplitTree(at, {})
        st.split_leaf(0)

        fid = 0
        sval = 6.32770014

        num_leafs = at.num_leafs();
        print("before: ", num_leafs)

        doms = [RealDomain() for i in range(at.num_features())]
        doms[fid].hi = sval
        at1 = at.prune(doms)
        doms[fid] = RealDomain()
        doms[fid].lo = sval
        at2 = at.prune(doms)

        print(at1.num_leafs(), at2.num_leafs(),
                num_leafs - at1.num_leafs() + num_leafs - at2.num_leafs())

        for i in range(len(at)):
            p = TreePlot()
            p.g.attr(label=f"X{fid} split at {sval}")
            p.add_splittree_leaf(at[i], st.get_leaf(1))
            p.add_splittree_leaf(at[i], st.get_leaf(2))
            p.render(f"/tmp/plots/test2-{i}")

if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
