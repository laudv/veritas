import os
import unittest
import pickle
import numpy as np

from veritas import AddTree, AddTreeType, FloatT, Interval, LtSplit
from veritas import TRUE_INTERVAL, FALSE_INTERVAL, BOOL_SPLIT_VALUE

BPATH = os.path.dirname(__file__)

class TestTree(unittest.TestCase):

    def myAssertAlmostEqual(self, a, b, eps=1e-6):
        self.assertTrue(type(a) == type(b))
        if isinstance(a, list) or isinstance(a, tuple) or isinstance(a, np.ndarray):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.myAssertAlmostEqual(x, y, eps=eps)
        elif isinstance(a, float) or isinstance(a, FloatT) or isinstance(a, np.float64):
            self.assertAlmostEqual(a, b, delta=eps)
        else:
            self.assertEqual(a, b)

    def test_tree1(self):
        at = AddTree(2, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 1, 16.0)
        t.set_leaf_value(t.left(t.root()), 0, 1.1)
        t.set_leaf_value(t.right(t.root()), 0, 2.2)
        t.set_leaf_value(t.left(t.root()), 1, -1.1)
        t.set_leaf_value(t.right(t.root()), 1, -2.2)

        y = at.eval(np.array([[1.0, 1.0, 3.0], [1.0, 22.0, 3.0]],
                             dtype=FloatT))

        self.myAssertAlmostEqual(np.array([[1.1, -1.1], [2.2, -2.2]],
                                          dtype=FloatT), y)

        self.assertRaises(RuntimeError, at.compute_box, [1, 2])
        self.assertEqual(at.compute_box([1]), {1: Interval.from_hi(16.0)})
        self.assertEqual(at.compute_box([2]), {1: Interval.from_lo(16.0)})

    def test_boolsplit(self):
        at = AddTree(2, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 0, 2.0)
        t.split(t.left(t.root()), 1, 1.0)
        t.split(t.right(t.root()), 2)
        t.set_leaf_value(t.left(t.left(t.root())), 0, 1.0)
        t.set_leaf_value(t.right(t.left(t.root())), 0, 2.0)
        t.set_leaf_value(t.left(t.right(t.root())), 0, 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 0, 8.0)
        t.set_leaf_value(t.left(t.left(t.root())), 1, -1.0)
        t.set_leaf_value(t.right(t.left(t.root())), 1, -2.0)
        t.set_leaf_value(t.left(t.right(t.root())), 1, -4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 1, -8.0)

        #print(at[0])

        self.assertEqual(t.get_split(        t.root() ), LtSplit(0, 2.0))
        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), LtSplit(2, BOOL_SPLIT_VALUE))

        self.assertEqual(at.compute_box([5]), {0: Interval.from_lo(2.0),
                                               2: FALSE_INTERVAL})
        self.assertEqual(at.compute_box([6]), {0: Interval.from_lo(2.0),
                                               2: TRUE_INTERVAL})

        T, F = 1.0, 0.0

        y = at.eval(np.array([
            [0.0, 0.5, F], [0.0, 1.5, T],
            [2.5, 0.5, T], [2.5, 0.5, F]], dtype=FloatT))

        self.myAssertAlmostEqual(y, np.array([[1.0, -1.0],
                                              [2.0, -2.0],
                                              [8.0, -8.0],
                                              [4.0, -4.0]], dtype=FloatT))

        s = at.to_json()
        att = AddTree.from_json(s)
        t = att[0]

        self.assertEqual(t.get_split( t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), LtSplit(2, BOOL_SPLIT_VALUE))

    def test_tree_json(self):
        at = AddTree(3, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.split(t.left(t.root()), 2, 4.0)
        t.set_leaf_value(t.left(t.left(t.root())), 0, 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0, 0.45)
        t.set_leaf_value(t.right(t.root()), 0, 2.2)
        t.set_leaf_value(t.left(t.left(t.root())), 1, -0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 1, -0.45)
        t.set_leaf_value(t.right(t.root()), 1, -2.2)

        self.assertEqual(at.compute_box([2]), {1: Interval.from_lo(2.0)})
        self.assertEqual(at.compute_box([3]), {1: Interval.from_hi(2.0),
                                               2: Interval.from_hi(4.0)})
        self.assertEqual(at.compute_box([4]), {1: Interval.from_hi(2.0),
                                               2: Interval.from_lo(4.0)})

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
        self.assertAlmostEqual(tt.get_leaf_value(2, 0), 2.2)
        self.assertAlmostEqual(tt.get_leaf_value(3, 0), 0.25)
        self.assertAlmostEqual(tt.get_leaf_value(4, 0), 0.45)
        self.assertAlmostEqual(tt.get_leaf_value(2, 1), -2.2)
        self.assertAlmostEqual(tt.get_leaf_value(3, 1), -0.25)
        self.assertAlmostEqual(tt.get_leaf_value(4, 1), -0.45)
        self.assertAlmostEqual(tt.get_leaf_value(2, 2), 0.0)
        self.assertAlmostEqual(tt.get_leaf_value(3, 2), 0.0)
        self.assertAlmostEqual(tt.get_leaf_value(4, 2), 0.0)
        self.assertTrue(np.all(tt.get_leaf_values(2) == np.array([2.2, -2.2, 0.0])))
        self.assertTrue(np.all(tt.get_leaf_values(3) == np.array([0.25, -0.25, 0.0])))
        self.assertTrue(np.all(tt.get_leaf_values(4) == np.array([0.45, -0.45, 0.0])))

    def test_addtree_get_splits(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 1, 4.0)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0, 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0, 0.45)
        t.set_leaf_value(t.right(t.root()), 0, 2.2)

        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.set_leaf_value(t.left(t.root()), 0, 0.5)
        t.set_leaf_value(t.right(t.root()), 0, 2.3)

        self.assertRaises(RuntimeError, at.compute_box, [2, 1]) # incompatible leafs
        self.assertEqual(at.compute_box([2, 2]), {1: Interval.from_lo(4.0)})
        self.assertEqual(at.compute_box([3, 1]), {1: Interval.from_hi(2.0),
                                                  2: Interval.from_hi(0.12)})

        s = at.get_splits()

        self.myAssertAlmostEqual(s[1], [2.0, 4.0])
        self.myAssertAlmostEqual(s[2], [0.12])
        self.assertEqual(sorted(list(s.keys())), [1, 2])

    def test_pickle1(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 1, 4.0)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0, 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0, 0.45)
        t.set_leaf_value(t.right(t.root()), 0, 2.2)

        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.set_leaf_value(t.left(t.root()), 0, 0.5)
        t.set_leaf_value(t.right(t.root()), 0, 2.3)

        att = pickle.loads(pickle.dumps(at))
        self.assertEqual(at.num_nodes(), att.num_nodes())

    def test_predict_xgb(self):
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        X = np.array([[x, y] for x in range(100) for y in range(100)], dtype=FloatT)

        self.assertTrue(at.get_type(), AddTreeType.CLF_SOFTMAX)

        ypred = at.eval(X)

        for c in range(at.num_leaf_values()):
            at0 = at.make_singleclass(c)
            self.assertTrue(at0.get_type() == AddTreeType.CLF_SOFTMAX)

            ypred0 = 1.0 / (1.0 + np.exp(-ypred[:, c]))
            ypred1 = at0.predict(X).ravel()
            ypred2 = 1.0 / (1.0 + np.exp(-at0.eval(X).ravel()))

            #print(ypred0)
            #print(ypred1)
            #print(ypred2)

            self.assertAlmostEqual(np.sum(np.abs(ypred0 - ypred1)), 0.0)
            self.assertAlmostEqual(np.sum(np.abs(ypred0 - ypred2)), 0.0)

    def test_predict_rf(self):
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        X = np.array([[x, y] for x in range(100) for y in range(100)], dtype=FloatT)

        self.assertTrue(at.get_type(), AddTreeType.CLF_MEAN)

        ypred = at.eval(X)

        for c in range(at.num_leaf_values()):
            at0 = at.make_singleclass(c)
            self.assertTrue(at0.get_type() == AddTreeType.CLF_MEAN)

            ypred0 = ypred[:, c] / len(at) + 0.5
            ypred1 = at0.predict(X).ravel()
            ypred2 = at0.eval(X).ravel() / len(at0) + 0.5

            #print(ypred0)
            #print(ypred1)
            #print(ypred2)

            self.assertAlmostEqual(np.sum(np.abs(ypred0 - ypred1)), 0.0)
            self.assertAlmostEqual(np.sum(np.abs(ypred0 - ypred2)), 0.0)


    def test_get_maximum_feat_id(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 3, 4.0)
         
        # check that get_maximum_feat_id returns 3
        self.assertEqual(at[0].get_maximum_feat_id(), 3)

    def test_min_num_columns(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 3, 4.0)

        # 2-column input (= too small)
        X = np.array([[0.0, 0.0]], dtype=FloatT)

        # both eval and predict should now fail (too few columns)
        self.assertRaises(RuntimeError, at[0].eval, X)
        self.assertRaises(RuntimeError, at.predict, X)
        self.assertRaises(RuntimeError, at.eval, X)
   
        

if __name__ == "__main__":
    unittest.main()
