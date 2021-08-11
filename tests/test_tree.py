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
        #print(y)

        self.myAssertAlmostEqual(np.array([1.1, 2.2], dtype=np.float32), y)

        self.assertRaises(RuntimeError, at.compute_box, [1, 2])
        self.assertEqual(at.compute_box([1]), {1: Domain.from_hi_exclusive(16.0)})
        self.assertEqual(at.compute_box([2]), {1: Domain.from_lo(16.0)})

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

        #print(at[0])

        self.assertEqual(t.get_split(        t.root() ), LtSplit(0, 2.0))
        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), LtSplit(2, BOOL_SPLIT_VALUE))

        self.assertEqual(at.compute_box([5]), {0: Domain.from_lo(2.0), 2: FALSE_DOMAIN})
        self.assertEqual(at.compute_box([6]), {0: Domain.from_lo(2.0), 2: TRUE_DOMAIN})

        T, F = 1.0, 0.0

        y = at.eval(np.array([
            [0.0, 0.5, F], [0.0, 1.5, T],
            [2.5, 0.5, T], [2.5, 0.5, F]], dtype=np.float32))

        self.myAssertAlmostEqual(y, np.array([1.0, 2.0, 8.0, 4.0], dtype=np.float32))

        s = at.to_json();
        att = AddTree.from_json(s)
        t = att[0]

        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), LtSplit(2, BOOL_SPLIT_VALUE))

    def test_tree_json(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.split(t.left(t.root()), 2, 4.0)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)

        self.assertEqual(at.compute_box([2]), {1: Domain.from_lo(2.0)})
        self.assertEqual(at.compute_box([3]), {1: Domain.from_hi_exclusive(2.0), 2: Domain.from_hi_exclusive(4.0)})
        self.assertEqual(at.compute_box([4]), {1: Domain.from_hi_exclusive(2.0), 2: Domain.from_lo(4.0)})

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

        self.assertRaises(RuntimeError, at.compute_box, [2, 1]) # incompatible leafs
        self.assertEqual(at.compute_box([2, 2]), {1: Domain.from_lo(4.0)})
        self.assertEqual(at.compute_box([3, 1]), {1: Domain.from_hi_exclusive(2.0),
                                                  2: Domain.from_hi_exclusive(0.12)})

        s = at.get_splits()

        self.myAssertAlmostEqual(s[1], [2.0, 4.0])
        self.myAssertAlmostEqual(s[2], [0.12])
        self.assertEqual(sorted(list(s.keys())), [1, 2])

    def test_pickle1(self):
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

        att = pickle.loads(pickle.dumps(at))
        self.assertEqual(at.num_nodes(), att.num_nodes())

if __name__ == "__main__":
    unittest.main()
