import unittest, pickle
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

        #print(at)

        st = SplitTree(at, {})
        l0 = st.get_leaf(0)
        l0.find_best_domtree_split(at)
        self.assertEqual(l0.get_best_split(), (0, 2))
        self.assertEqual(l0.split_score, 4)   # 2 leafs left, 2 right
        self.assertEqual(l0.split_balance, 0) # perfectly balanced
        m, M = l0.get_tree_bounds(at, 0)
        self.assertEqual(m, 0.1)
        self.assertEqual(M, 0.4)
        st.split(l0)

        l1 = st.get_leaf(1)
        self.assertTrue(l1.is_reachable(0, 0))
        self.assertTrue(l1.is_reachable(0, 1))
        self.assertTrue(l1.is_reachable(0, 3))
        self.assertTrue(l1.is_reachable(0, 4))
        self.assertFalse(l1.is_reachable(0, 2))
        self.assertFalse(l1.is_reachable(0, 5))
        self.assertFalse(l1.is_reachable(0, 6))
        m, M = l1.get_tree_bounds(at, 0)
        self.assertEqual(m, 0.1)
        self.assertEqual(M, 0.2)

        l2 = st.get_leaf(2)
        self.assertTrue(l2.is_reachable(0, 0))
        self.assertFalse(l2.is_reachable(0, 1))
        self.assertFalse(l2.is_reachable(0, 3))
        self.assertFalse(l2.is_reachable(0, 4))
        self.assertTrue(l2.is_reachable(0, 2))
        self.assertTrue(l2.is_reachable(0, 5))
        self.assertTrue(l2.is_reachable(0, 6))
        m, M = l2.get_tree_bounds(at, 0)
        self.assertEqual(m, 0.3)
        self.assertEqual(M, 0.4)

        # with root_domain
        st = SplitTree(at, {0: RealDomain(0, 2)})
        self.assertEqual(st.get_root_domain(0), RealDomain(0, 2))
        l0 = st.get_leaf(0)
        self.assertTrue(l0.is_reachable(0, 0))
        self.assertTrue(l0.is_reachable(0, 1))
        self.assertTrue(l0.is_reachable(0, 3))
        self.assertTrue(l0.is_reachable(0, 4))
        self.assertFalse(l0.is_reachable(0, 2))
        self.assertFalse(l0.is_reachable(0, 5))
        self.assertFalse(l0.is_reachable(0, 6))
        m, M = l0.get_tree_bounds(at, 0)
        self.assertEqual(m, 0.1)
        self.assertEqual(M, 0.2)

        l0.find_best_domtree_split(at)
        self.assertEqual(l0.get_best_split(), (0, 1))
        self.assertEqual(l0.split_score, 2)
        self.assertEqual(l0.split_balance, 0)
        st.split(l0)

        self.assertEqual(st.get_leaf_domains(1), {0: RealDomain(0, 1)})
        self.assertEqual(st.get_leaf_domains(2), {0: RealDomain(1, 2)})

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

        #print(at)

        st = SplitTree(at, {})
        l0 = st.get_leaf(0)
        l0.find_best_domtree_split(at)
        self.assertEqual(l0.get_best_split(), (0, 3.0))
        self.assertEqual(l0.split_score, 9)
        self.assertEqual(l0.split_balance, 1)
        l0.mark_unreachable(1, 2)
        l0.find_best_domtree_split(at)
        self.assertEqual(l0.get_best_split(), (0, 3.0))
        self.assertEqual(l0.split_score, 6)
        self.assertEqual(l0.split_balance, 4)

        l0 = st.get_leaf(0)
        st.split(l0)

        def test_l1(l1):
            self.assertFalse(l1.is_reachable(0, 6))
            self.assertFalse(l1.is_reachable(1, 2))
            self.assertFalse(l1.is_reachable(1, 5))
            self.assertFalse(l1.is_reachable(1, 6))
            self.assertFalse(l1.is_reachable(1, 7))
            self.assertFalse(l1.is_reachable(1, 8))
            self.assertTrue(l1.is_reachable(0, 0))
            self.assertTrue(l1.is_reachable(0, 1))
            self.assertTrue(l1.is_reachable(1, 1))

        l1 = st.get_leaf(1)
        test_l1(l1)

        def test_l2(l2):
            self.assertFalse(l2.is_reachable(0, 1))
            self.assertFalse(l2.is_reachable(0, 3))
            self.assertFalse(l2.is_reachable(0, 4))
            self.assertFalse(l2.is_reachable(0, 5))
            self.assertFalse(l2.is_reachable(1, 1))
            self.assertFalse(l2.is_reachable(1, 3))
            self.assertFalse(l2.is_reachable(1, 4))
            self.assertTrue(l2.is_reachable(0, 0))
            self.assertTrue(l2.is_reachable(0, 6))
            self.assertTrue(l2.is_reachable(1, 2))

        l2 = st.get_leaf(2)
        test_l2(l2)

        x = pickle.dumps(l1)
        l1c = pickle.loads(pickle.dumps(l1))
        test_l1(l1c)

        x = pickle.dumps(l2)
        l2c = pickle.loads(pickle.dumps(l2))
        test_l2(l2c)

        stt = SplitTree.from_json(at, st.to_json())
        l1t = stt.get_leaf(1)
        test_l1(l1t)
        l2t = stt.get_leaf(2)
        test_l2(l2t)

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-hard.json")
        st = SplitTree(at, {})
        l0 = st.get_leaf(0)
        l0.find_best_domtree_split(at)

        fid, sval = l0.get_best_split()
        split_score = l0.split_score
        split_balance = l0.split_balance
        
        st.split(l0)

        num_leafs = at.num_leafs();

        doms_l = [RealDomain() for i in range(at.num_features())]
        doms_l[fid].hi = sval
        doms_r = [RealDomain() for i in range(at.num_features())]
        doms_r[fid].lo = sval
        at_l = at.prune(doms_l)
        at_r = at.prune(doms_r)

        num_leafs_l = at_l.num_leafs()
        num_leafs_r = at_r.num_leafs()
        score = num_leafs - num_leafs_l + num_leafs - num_leafs_r
        balance = abs(num_leafs_l - num_leafs_r)
        self.assertEqual(score, split_score)
        self.assertEqual(balance, split_balance)

        #for i in range(len(at)):
        #    p = TreePlot()
        #    p.g.attr(label=f"X{fid} split at {sval}")
        #    p.add_splittree_leaf(at[i], st.get_leaf(1))
        #    p.add_splittree_leaf(at[i], st.get_leaf(2))
        #    p.render(f"/tmp/plots/test2-{i}")

if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
