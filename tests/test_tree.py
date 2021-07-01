import unittest, pickle, math
import numpy as np
from veritas import *

class TestTree(unittest.TestCase):

    def myAssertAlmostEqual(self, a, b, eps=1e-6):
        self.assertTrue(type(a) == type(b))
        if isinstance(a, list) or isinstance(a, tuple) or isinstance(a, np.ndarray):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.myAssertAlmostEqual(x, y, eps=eps)
        elif isinstance(a, float) or isinstance(a, np.float32) or isinstance(a, np.float64):
            self.assertAlmostEqual(a, b, delta=eps)
        else:
            self.assertEqual(a, b)

    def test_tree1(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 16.0)
        t.set_leaf_value(t.left(t.root()), 1.1)
        t.set_leaf_value(t.right(t.root()), 2.2)

        y = at.eval(np.array([[1.0, 1.0, 3.0], [1.0, 22.0, 3.0]], dtype=np.float32))
        print(y)

        self.myAssertAlmostEqual(np.array([1.1, 2.2], dtype=np.float32), y)

        self.assertRaises(RuntimeError, at.compute_box, [1, 2])
        self.assertEqual(at.compute_box([1]), {1: Domain.from_hi_exclusive(16.0)})
        self.assertEqual(at.compute_box([2]), {1: Domain.from_lo(16.0)})

    def test_tree_json(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.split(t.left(t.root()), 2, 4.0)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)

        self.assertEqual(at.get_domains([2]), {1: RealDomain(2.0, math.inf)})
        self.assertEqual(at.get_domains([3]), {1: RealDomain(-math.inf, 2.0), 2: RealDomain(-math.inf, 4.0)})
        self.assertEqual(at.get_domains([4]), {1: RealDomain(-math.inf, 2.0), 2: RealDomain(4.0, math.inf)})

        s = at.to_json();
        att = AddTree.from_json(s)
        tt = att[0]

        self.assertTrue(tt.is_internal(0))
        self.assertTrue(tt.is_internal(1))
        self.myAssertAlmostEqual(tt.get_split(0), LtSplit(1, 2.0))
        self.myAssertAlmostEqual(tt.get_split(1), LtSplit(2, 4.0))
        self.assertTrue(tt.is_leaf(2))
        self.assertTrue(tt.is_leaf(3))
        self.assertTrue(tt.is_leaf(4))
        self.assertAlmostEqual(tt.get_leaf_value(2), 2.2)
        self.assertAlmostEqual(tt.get_leaf_value(3), 0.25)
        self.assertAlmostEqual(tt.get_leaf_value(4), 0.45)

    def test_boolsplit(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 0, 2.0)
        t.split(t.left(t.root()), 1, 1.0)
        t.split(t.right(t.root()), 2)
        t.set_leaf_value(t.left(t.left(t.root())), 1.0)
        t.set_leaf_value(t.right(t.left(t.root())), 2.0)
        t.set_leaf_value(t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        #print(at)

        self.assertEqual(t.get_split(        t.root() ), LtSplit(0, 2.0))
        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), BoolSplit(2))

        self.assertEqual(at.get_domains([5]), {0: RealDomain(2.0, math.inf), 2: BoolDomain(True)})
        self.assertEqual(at.get_domains([6]), {0: RealDomain(2.0, math.inf), 2: BoolDomain(False)})

        y = at.predict([
            [0.0, 0.5, True], [0.0, 1.5, True],
            [2.5, 0.5, True], [2.5, 0.5, False]])

        self.assertEqual(y, [1.0, 2.0, 4.0, 8.0])

        types = AddTreeFeatureTypes(at)
        self.assertEqual(types[0], LtSplit)
        self.assertEqual(types[1], LtSplit)
        self.assertEqual(types[2], BoolSplit)

        s = at.to_json();
        att = AddTree.from_json(s)
        t = att[0]

        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), BoolSplit(2))

    def test_at_feature_types(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 0, 1.5)    # lt split
        t.split(t.left(t.root()), 0) # bool split
        self.assertRaises(Exception, AddTreeFeatureTypes, at)

    def test_addtree_get_splits(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 4.0)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)

        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.set_leaf_value(t.left(t.root()), 0.5)
        t.set_leaf_value(t.right(t.root()), 2.3)

        #print(at)

        self.assertRaises(RuntimeError, at.get_domains, [2, 1]) # incompatible leafs
        self.assertEqual(at.get_domains([2, 2]), {1: RealDomain(4.0, math.inf)})

        s = at.get_splits()

        self.myAssertAlmostEqual(s[1], [2.0, 4.0])
        self.myAssertAlmostEqual(s[2], [0.12])
        self.assertEqual(sorted(list(s.keys())), [1, 2])

    def test_skip_branch_left(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.set_leaf_value(t.left(t.root()), 1.1)
        t.set_leaf_value(t.right(t.root()), 2.2)

        self.assertEqual(t.tree_size(t.root()), 3)
        self.assertTrue(t.is_internal(t.root()))
        self.assertEqual(t.depth(t.right(t.root())), 1)

        t.skip_branch(t.left(t.root()))

        self.assertEqual(t.tree_size(t.root()), 1)
        self.assertTrue(t.is_leaf(t.root()))

    def test_skip_branch_right(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)
        self.assertEqual(t.tree_size(t.root()), 5)
        self.assertTrue(t.is_internal(t.root()))

        t.skip_branch(t.right(t.root()))

        self.assertEqual(t.tree_size(t.root()), 3)
        self.assertTrue(t.is_internal(t.root()))
        self.assertEqual(t.depth(t.right(t.root())), 1)
        self.assertEqual(t.depth(t.left(t.root())), 1)

    def test_skip_branch_internal(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)
        self.assertEqual(t.tree_size(t.root()), 5)
        self.assertTrue(t.is_internal(t.root()))

        print("OLD TREE\n", t);

        t.skip_branch(t.left(t.left(t.root())))

        print("NEW TREE\n", t);

        self.assertEqual(t.tree_size(t.root()), 3)
        self.assertTrue(t.is_internal(t.root()))
        self.assertTrue(t.is_leaf(t.left(t.root())))
        self.assertTrue(t.is_leaf(t.right(t.root())))
        self.assertAlmostEqual(t.get_leaf_value(t.left(t.root())), 0.45)
        self.assertAlmostEqual(t.get_leaf_value(t.right(t.root())), 2.2)

    def test_pickle(self):
        at = AddTree.read("tests/models/xgb-covtype-easy.json")
        att = pickle.loads(pickle.dumps(at))
        self.assertEqual(at.num_nodes(), att.num_nodes())

if __name__ == "__main__":
    unittest.main()
