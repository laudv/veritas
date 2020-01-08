import math
import unittest, pickle
import numpy as np

from treeck import *
from treeck.plot import TreePlot

class TestDomTree(unittest.TestCase):
    def test_single_tree(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        print(at)

        dt = DomTree(at, {})
        self.assertEqual(1, dt.num_instances())
        self.assertTrue(dt.get_root_domain(0, 0) is None)
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        self.assertEqual(l0.get_best_split(), (0, "lt", 0, 2))
        self.assertEqual(l0.score, 4)   # 2 leafs left, 2 right
        self.assertEqual(l0.balance, 0) # perfectly balanced
        m, M = l0.get_tree_bounds(0, 0)
        self.assertEqual(m, 1.0)
        self.assertEqual(M, 8.0)
        dt.apply_leaf(l0)

        l1 = dt.get_leaf(1)
        self.assertEqual(l1.get_domains(0), {0: RealDomain(-math.inf, 2)})
        self.assertEqual(l1.get_domain(0, 0), RealDomain(-math.inf, 2))
        self.assertTrue( l1.is_reachable(0, 0, 0))
        self.assertTrue( l1.is_reachable(0, 0, 1))
        self.assertTrue( l1.is_reachable(0, 0, 3))
        self.assertTrue( l1.is_reachable(0, 0, 4))
        self.assertFalse(l1.is_reachable(0, 0, 2))
        #self.assertFalse(l1.is_reachable(0, 0, 5)) // unreachable because 0, 2 blocked
        #self.assertFalse(l1.is_reachable(0, 0, 6)) // ^
        m, M = l1.get_tree_bounds(0, 0)
        self.assertEqual(m, 1.0)
        self.assertEqual(M, 2.0)

        l2 = dt.get_leaf(2)
        self.assertEqual(l2.get_domains(0), {0: RealDomain(2, math.inf)})
        self.assertEqual(l2.get_domain(0, 0), RealDomain(2, math.inf))
        self.assertTrue( l2.is_reachable(0, 0, 0))
        self.assertFalse(l2.is_reachable(0, 0, 1))
        #self.assertFalse(l2.is_reachable(0, 0, 3)) // unreachable because 0, 1 blocked
        #self.assertFalse(l2.is_reachable(0, 0, 4)) // ^
        self.assertTrue( l2.is_reachable(0, 0, 2))
        self.assertTrue( l2.is_reachable(0, 0, 5))
        self.assertTrue( l2.is_reachable(0, 0, 6))
        m, M = l2.get_tree_bounds(0, 0)
        self.assertEqual(m, 4.0)
        self.assertEqual(M, 8.0)

        # with root_domain
        dt = DomTree(at, {0: RealDomain(0, 2)})
        self.assertEqual(dt.get_root_domain(0, 0), RealDomain(0, 2))
        l0 = dt.get_leaf(0)
        self.assertEqual(l0.get_domains(0), {0: RealDomain(0, 2)})
        self.assertEqual(l0.get_domain(0, 0), dt.get_root_domain(0, 0))
        self.assertTrue( l0.is_reachable(0, 0, 0))
        self.assertTrue( l0.is_reachable(0, 0, 1))
        self.assertTrue( l0.is_reachable(0, 0, 3))
        self.assertTrue( l0.is_reachable(0, 0, 4))
        self.assertFalse(l0.is_reachable(0, 0, 2))
        #self.assertTrue(l0.is_reachable(0, 0, 5)) # hidden by 0, 2
        #self.assertTrue(l0.is_reachable(0, 0, 6))
        m, M = l0.get_tree_bounds(0, 0)
        self.assertEqual(m, 1.0)
        self.assertEqual(M, 2.0)

        l0.find_best_split()
        self.assertEqual(l0.get_best_split(), (0, "lt", 0, 1))
        self.assertEqual(l0.score, 2)
        self.assertEqual(l0.balance, 0)
        dt.apply_leaf(l0)

        self.assertEqual(dt.get_leaf(1).get_domains(0), {0: RealDomain(0, 1)})
        self.assertEqual(dt.get_leaf(2).get_domains(0), {0: RealDomain(1, 2)})

    def test_serialize(self):
        at = AddTree.read("tests/models/xgb-calhouse-hard.json")

        dt = DomTree(at, {})
        self.assertEqual(dt.num_instances(), 1)
        l0 = dt.get_leaf(0);
        nbytes1 = len(pickle.dumps(l0))
        print("num of kb", nbytes1)

        dt = DomTree([(at, {}), (at, {0: RealDomain(0, 1)})])
        self.assertEqual(dt.num_instances(), 2)
        l0 = dt.get_leaf(0);
        nbytes2 = len(pickle.dumps(l0))
        print("num of kb", nbytes2)
        self.assertLess(nbytes2, 2*nbytes1)

        l0c = pickle.loads(pickle.dumps(l0))
        self.assertEqual(l0c.num_instances(), 2)
        self.assertEqual(l0.get_domains(0), {})
        self.assertEqual(l0c.get_domains(0), {})
        self.assertNotEqual(l0.get_domains(1), {})
        self.assertNotEqual(l0c.get_domains(1), {})
        self.assertEqual(l0c.get_domains(1), l0.get_domains(1))
        self.assertEqual(nbytes2, len(pickle.dumps(l0c)))

    def test_get_domains1(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 1) # bool split
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        dt = DomTree(at, {0: RealDomain(0, 2)})
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        dt.apply_leaf(l0)

        self.assertEqual(dt.get_leaf(1).get_domains(0), {0: RealDomain(0, 1)})
        self.assertEqual(dt.get_leaf(2).get_domains(0), {0: RealDomain(1, 2)})

    def test_get_domains2(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0)
        t.split( t.left(t.root()), 1, 2) # bool split
        t.split(t.right(t.root()), 1, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        dt = DomTree(at, {})
        l0 = dt.get_leaf(0);
        l0.find_best_split();
        self.assertEqual(l0.get_best_split(), (0,"bool", 0))
        dt.apply_leaf(l0)

        self.assertEqual(dt.get_leaf(1).get_domains(0), {0: BoolDomain(False)})
        self.assertEqual(dt.get_leaf(2).get_domains(0), {0: BoolDomain(True)})

    def test_two_trees(self):
        at = AddTree()
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
        t.split(t.right(t.right(t.root())), 2)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value( t.left(t.right(t.right(t.root()))), 0.5)
        t.set_leaf_value(t.right(t.right(t.right(t.root()))), 0.6)

        print(at)

        dt = DomTree(at, {})
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        self.assertEqual(l0.get_best_split(), (0, "lt", 0, 3.0))
        self.assertEqual(l0.score, 9)
        self.assertEqual(l0.balance, 1)
        l0.mark_unreachable(0, 1, 2)
        l0.find_best_split()
        self.assertEqual(l0.get_best_split(), (0, "lt", 0, 3.0))
        self.assertEqual(l0.score, 6)
        self.assertEqual(l0.balance, 4)

        l0 = dt.get_leaf(0)
        l0.find_best_split()
        dt.apply_leaf(l0)

        def test_l1(l1):
            self.assertFalse(l1.is_reachable(0, 0, 6))
            self.assertFalse(l1.is_reachable(0, 1, 2))
            #self.assertFalse(l1.is_reachable(0, 1, 5)) # hidden by 1, 2
            #self.assertFalse(l1.is_reachable(0, 1, 6)) # ^
            #self.assertFalse(l1.is_reachable(0, 1, 7)) # ^
            #self.assertFalse(l1.is_reachable(0, 1, 8)) # ^
            self.assertTrue( l1.is_reachable(0, 0, 0))
            self.assertTrue( l1.is_reachable(0, 0, 1))
            self.assertTrue( l1.is_reachable(0, 1, 1))

        l1 = dt.get_leaf(1)
        test_l1(l1)

        def test_l2(l2):
            self.assertFalse(l2.is_reachable(0, 0, 1))
            #self.assertFalse(l2.is_reachable(0, 0, 3)) # hidden by 0, 1
            #self.assertFalse(l2.is_reachable(0, 0, 4)) # ^
            self.assertFalse(l2.is_reachable(0, 0, 5))
            self.assertFalse(l2.is_reachable(0, 1, 1))
            #self.assertFalse(l2.is_reachable(0, 1, 3)) # hidden by 1, 1
            #self.assertFalse(l2.is_reachable(0, 1, 4)) # ^
            self.assertTrue( l2.is_reachable(0, 0, 0))
            self.assertTrue( l2.is_reachable(0, 0, 6))
            self.assertTrue( l2.is_reachable(0, 1, 2))

        l2 = dt.get_leaf(2)
        test_l2(l2)

        x = pickle.dumps(l1)
        l1c = pickle.loads(pickle.dumps(l1))
        test_l1(l1c)

        x = pickle.dumps(l2)
        l2c = pickle.loads(pickle.dumps(l2))
        test_l2(l2c)

        self.assertEqual(l1c.get_domains(0), l1.get_domains(0))
        self.assertEqual(l2c.get_domains(0), l2.get_domains(0))

    def test_merge(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 1)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        dt = DomTree([(at, {}), (at, {})])
        l0_0 = dt.get_leaf(0)
        l0_1 = dt.get_leaf(0)

        l0_0.mark_unreachable(0, 0, 1)
        l0_0.mark_unreachable(1, 0, 2)
        self.assertFalse(l0_0.is_reachable(0, 0, 1))
        self.assertTrue( l0_0.is_reachable(0, 0, 2))
        self.assertFalse(l0_0.is_reachable(1, 0, 2))
        self.assertTrue( l0_0.is_reachable(1, 0, 1))
        l0_1.mark_unreachable(0, 0, 2)
        l0_1.mark_unreachable(1, 0, 1)
        self.assertFalse(l0_1.is_reachable(0, 0, 2))
        self.assertTrue( l0_1.is_reachable(0, 0, 1))
        self.assertFalse(l0_1.is_reachable(1, 0, 1))
        self.assertTrue( l0_1.is_reachable(1, 0, 2))

        l0_m = DomTreeLeaf.merge([l0_0, l0_1]);
        self.assertFalse(l0_m.is_reachable(0, 0, 2))
        self.assertFalse(l0_m.is_reachable(0, 0, 1))
        self.assertFalse(l0_m.is_reachable(1, 0, 2))
        self.assertFalse(l0_m.is_reachable(1, 0, 1))

    def test_multi_instance(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 1)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        dt = DomTree([(at, {0: RealDomain(0, 1)}), (at, {})])
        self.assertEqual(dt.num_instances(), 2)
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        self.assertEqual(l0.get_best_split(), (1, "lt", 0, 2.0))
        self.assertEqual(l0.score, 4)
        self.assertEqual(l0.balance, 0)

    #def _test_calhouse(self): # Prune removed
    #    at = AddTree.read("tests/models/xgb-calhouse-hard.json")
    #    dt = DomTree(at, {})
    #    l0 = dt.get_leaf(0)
    #    l0.find_best_split()

    #    _, fid, sval = l0.get_best_split()
    #    split_score = l0.score
    #    split_balance = l0.balance
    #    
    #    dt.split(l0)

    #    num_leafs = at.num_leafs();
    #    ftypes = AddTreeFeatureTypes(at)

    #    doms_l = [RealDomain() for i in ftypes.feat_ids()]
    #    doms_l[fid].hi = sval
    #    doms_r = [RealDomain() for i in ftypes.feat_ids()]
    #    doms_r[fid].lo = sval
    #    at_l = at.prune(doms_l)
    #    at_r = at.prune(doms_r)

    #    num_leafs_l = at_l.num_leafs()
    #    num_leafs_r = at_r.num_leafs()
    #    score = num_leafs - num_leafs_l + num_leafs - num_leafs_r
    #    balance = abs(num_leafs_l - num_leafs_r)
    #    self.assertEqual(score, split_score)
    #    self.assertEqual(balance, split_balance)

    #    #for i in range(len(at)):
    #    #    p = TreePlot()
    #    #    p.g.attr(label=f"X{fid} split at {sval}")
    #    #    p.add_subspace(at[i], dt.get_leaf(1))
    #    #    p.add_subspace(at[i], dt.get_leaf(2))
    #    #    p.render(f"/tmp/plots/test2-{i}")

    def test_img_multisplit(self):
        def plt(at, ll, lr, instance, split_type, fid, sval):
            for i in range(len(at)):
                p = TreePlot()
                idl = ll.domtree_leaf_id()
                idr = lr.domtree_leaf_id()
                p.g.attr(label=f"X{fid} split at {sval} (ids={idl}, {idr})")
                p.add_domtree_leaf(instance, at[i], ll)
                p.add_domtree_leaf(instance, at[i], lr)
                p.render(f"/tmp/plots/multisplit-{idl}-{idr}-{i}")

        at = AddTree.read("tests/models/xgb-img-easy.json")
        dt = DomTree(at, {})
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        b0 = l0.get_best_split()
        print("l0", b0)

        dt.apply_leaf(l0)

        l1 = dt.get_leaf(1)
        l2 = dt.get_leaf(2)

        l1.find_best_split()
        l2.find_best_split()
        b1 = l1.get_best_split()
        b2 = l2.get_best_split()

        plt(at, l1, l2, *b0)

        print("l1", b1)
        print("l2", b2)

        self.assertNotEqual(b0, b1)
        self.assertNotEqual(b0, b2)

        dt.apply_leaf(l1)

        l3 = dt.get_leaf(3)
        l4 = dt.get_leaf(4)

        l3.find_best_split()
        l4.find_best_split()
        b3 = l3.get_best_split()
        b4 = l4.get_best_split()

        plt(at, l3, l4, *b1)

        print("l3", b3)
        print("l4", b4)

        self.assertNotEqual(b1, b3)
        self.assertNotEqual(b1, b4)

    def test_bin_mnist(self):
        at = AddTree.read("tests/models/xgb-mnist-bin-yis1-easy.json")
        dt = DomTree(at, {})
        l0 = dt.get_leaf(0)
        l0.find_best_split()
        b0 = l0.get_best_split()
        print("l0", b0)

        self.assertEqual(at[0].get_split(0), b0[1:])

        dt.apply_leaf(l0)

        l1 = dt.get_leaf(1)
        self.assertTrue(l1.is_reachable(0, 0, 1))
        self.assertFalse(l1.is_reachable(0, 0, 2))

if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
