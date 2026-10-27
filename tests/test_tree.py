import os
import pickle
import unittest

import numpy as np
from veritas import (
    BOOL_SPLIT_VALUE,
    FALSE_INTERVAL,
    TRUE_INTERVAL,
    AddTree,
    AddTreeType,
    FastEvalAddTree,
    FloatT,
    Interval,
    LtSplit,
)

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

        y = at.eval(np.array([[1.0, 1.0, 3.0], [1.0, 22.0, 3.0]], dtype=FloatT))

        self.myAssertAlmostEqual(np.array([[1.1, -1.1], [2.2, -2.2]], dtype=FloatT), y)

        self.assertRaises(RuntimeError, at.compute_box, [1, 2])
        self.assertEqual(at.compute_box([1]), {1: Interval.from_hi(16.0)})
        self.assertEqual(at.compute_box([2]), {1: Interval.from_lo(16.0)})

        self.assertEqual(t.depth(t.root()), 0)
        self.assertEqual(t.depth(t["l"]), 1)
        self.assertEqual(t.depth(t["r"]), 1)
        self.assertEqual(at.max_depth(), 1)
        self.assertEqual(t.max_depth(), 1)
        self.assertEqual(t.max_depth(t["l"]), 0)
        self.assertEqual(t.max_depth(t["r"]), 0)
        self.assertEqual(at.max_depth(), t.depth(t["l"]))

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

        # print(at[0])

        self.assertEqual(t.max_depth(), t.depth(t["ll"]))
        self.assertTrue(t.is_leaf(t["ll"]))

        self.assertEqual(t.get_split(t.root()), LtSplit(0, 2.0))
        self.assertEqual(t.get_split(t.left(t.root())), LtSplit(1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), LtSplit(2, BOOL_SPLIT_VALUE))

        self.assertEqual(at.compute_box([5]), {0: Interval.from_lo(2.0), 2: FALSE_INTERVAL})
        self.assertEqual(at.compute_box([6]), {0: Interval.from_lo(2.0), 2: TRUE_INTERVAL})

        T, F = 1.0, 0.0

        X = np.array([[0.0, 0.5, F], [0.0, 1.5, T], [2.5, 0.5, T], [2.5, 0.5, F]], dtype=FloatT)
        y = at.eval(X)

        self.myAssertAlmostEqual(y, np.array([[1.0, -1.0], [2.0, -2.0], [8.0, -8.0], [4.0, -4.0]], dtype=FloatT))

        fat = FastEvalAddTree(at)
        self.assertTrue(np.array_equal(fat.eval(X), y))
        self.assertEqual(fat.eval_node(X)[:, 0].tolist(), [t["ll"], t["lr"], t["rr"], t["rl"]])

        s = at.to_json()
        att = AddTree.from_json(s)
        t = att[0]

        self.assertEqual(t.get_split(t.left(t.root())), LtSplit(1, 1.0))
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
        self.assertEqual(at.compute_box([3]), {1: Interval.from_hi(2.0), 2: Interval.from_hi(4.0)})
        self.assertEqual(at.compute_box([4]), {1: Interval.from_hi(2.0), 2: Interval.from_lo(4.0)})

        s = at.to_json()
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

        self.assertRaises(RuntimeError, at.compute_box, [2, 1])  # incompatible leafs
        self.assertEqual(at.compute_box([2, 2]), {1: Interval.from_lo(4.0)})
        self.assertEqual(at.compute_box([3, 1]), {1: Interval.from_hi(2.0), 2: Interval.from_hi(0.12)})

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

            # print(ypred0)
            # print(ypred1)
            # print(ypred2)

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

            # print(ypred0)
            # print(ypred1)
            # print(ypred2)

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
        self.assertRaises(RuntimeError, FastEvalAddTree(at).eval, X)
        self.assertRaises(RuntimeError, FastEvalAddTree(at).eval_node, X)

    def test_eval_node(self):  # https://github.com/laudv/veritas/issues/11
        # One stump: split on feature 0 at 0.0 -> left leaf (id 1), right leaf (id 2)
        at = AddTree(1, AddTreeType.CLF_SOFTMAX)
        t = at.add_tree()
        t.split(t.root(), 0, 0.0)
        t.set_leaf_value(t.left(t.root()), 0, -1.0)
        t.set_leaf_value(t.right(t.root()), 0, +1.0)

        # feature0 = -1 must reach leaf 1 (left); feature0 = +1 must reach leaf 2 (right)
        X = np.ascontiguousarray(np.array([[-1.0], [1.0]], dtype=FloatT))

        # Test node ids returned by eval_node
        self.assertEqual(t.eval_node(np.ascontiguousarray(X[0:1]))[0], 1)
        self.assertEqual(t.eval_node(np.ascontiguousarray(X[1:2]))[0], 2)
        self.assertTrue(np.all(t.eval_node(X) == np.array([1, 2])))

        # Test leaf values returned by eval
        self.assertTrue(np.all(t.eval(X) == np.array([[-1.0], [+1.0]])))

        # FastEvalAddTree: split value itself goes right, the next float below it goes left
        fat = FastEvalAddTree(at)
        below = np.nextafter(FloatT(0.0), FloatT(-np.inf))
        Xb = np.array([[-1.0], [1.0], [below], [0.0], [-np.inf], [np.inf]], dtype=FloatT)
        self.assertEqual(t.eval_node(Xb).tolist(), [1, 2, 1, 2, 1, 2])
        self.assertEqual(fat.eval_node(Xb)[:, 0].tolist(), [1, 2, 1, 2, 1, 2])
        self.assertTrue(np.array_equal(fat.eval(Xb), at.eval(Xb)))
        self.assertEqual(fat.eval_node(X[1]).tolist(), [[2]])

    def assertFastEvalEqual(self, at, X):
        fat = FastEvalAddTree(at)
        self.assertTrue(np.array_equal(fat.eval(X), at.eval(X)))
        self.assertTrue(np.array_equal(fat.eval_node(X), np.stack([t.eval_node(X) for t in at], axis=1)))

    def test_fast_eval_models(self):
        # integer grid hits every split of the xgb-img models exactly, and lies next to the rf-img splits
        X = np.array([[x, y] for x in range(100) for y in range(100)], dtype=FloatT)
        for name in ["xgb-img-hard-new", "xgb-img-multiclass", "xgb-img-multiclass-multivalue", "rf-img-multiclass"]:
            with self.subTest(model=name):
                at = AddTree.read(os.path.join(BPATH, f"models/{name}.json"))
                self.assertFastEvalEqual(at, X)
                fat = FastEvalAddTree(at)
                self.assertEqual(len(fat), len(at))
                self.assertEqual(fat.num_nodes(), at.num_nodes())
                self.assertEqual(fat.num_leaf_values(), at.num_leaf_values())

    def test_fast_eval_layouts(self):
        # rf_model has 13 features: non-contiguous column access matters
        at = AddTree.read(os.path.join(BPATH, "models/rf_model.json"))
        rng = np.random.default_rng(1)
        X = np.zeros((500, at.get_maximum_feat_id() + 1), dtype=FloatT)
        for f, vals in at.get_splits().items():
            vals = np.array(vals, dtype=FloatT)
            X[:, f] = rng.choice(np.concatenate([vals, vals - 1.0, vals + 1.0]), size=X.shape[0])
        Xcopy = X.copy()

        fat = FastEvalAddTree(at)
        y = at.eval(X)
        nodes = np.stack([t.eval_node(X) for t in at], axis=1)

        Xwide = np.zeros((X.shape[0] * 2, X.shape[1] * 3), dtype=FloatT)
        Xwide[::2, ::3] = X
        for Xl, rows in [
            (X, slice(None)),
            (np.asfortranarray(X), slice(None)),
            (Xwide[::2, ::3], slice(None)),
            (X[::-1], slice(None, None, -1)),
            (Xwide[-2::-2, ::3], slice(None, None, -1)),
            (X[:, ::-1][:, ::-1], slice(None)),
        ]:
            self.assertTrue(np.array_equal(fat.eval(Xl), y[rows]))
            self.assertTrue(np.array_equal(fat.eval_node(Xl), nodes[rows]))
        self.assertTrue(np.array_equal(X, Xcopy))
        self.assertFastEvalEqual(at, X.astype(np.float32))

    def test_fast_eval_edge_cases(self):
        at = AddTree(2, AddTreeType.REGR)
        at.set_base_score(0, 0.5)
        at.set_base_score(1, -0.25)

        # empty ensemble: only base scores
        fat = FastEvalAddTree(at)
        self.assertEqual(fat.eval(np.zeros((2, 1))).tolist(), [[0.5, -0.25]] * 2)
        self.assertEqual(fat.eval_node(np.zeros((2, 1))).shape, (2, 0))

        # leaf-only tree
        t = at.add_tree()
        t.set_leaf_value(t.root(), 0, 1.0)
        self.assertFastEvalEqual(at, np.zeros((3, 2), dtype=FloatT))

        t = at.add_tree()
        t.split(t.root(), 0, 2.0)
        t.split(t.left(t.root()), 1, 1.0)
        t.set_leaf_value(t["ll"], 1, 3.0)
        t.set_leaf_value(t["r"], 0, -2.0)
        X = np.array([[0.0, 0.5], [0.0, 1.5], [3.0, 0.0]], dtype=FloatT)
        self.assertFastEvalEqual(at, X)

        fat = FastEvalAddTree(at)
        expected = [[0, t["ll"]], [0, t["lr"]], [0, t["r"]]]
        self.assertEqual(fat.eval_node(X).tolist(), expected)
        self.assertEqual(fat.eval_node(X.tolist()).tolist(), expected)
        self.assertEqual(fat.eval_node(X.astype(np.int32)).tolist(), expected)
        self.assertEqual(fat.eval(np.zeros((0, 2), dtype=FloatT)).shape, (0, 2))
        self.assertEqual(fat.eval_node(np.zeros((0, 2), dtype=FloatT)).shape, (0, 2))

        # snapshot: later modifications to the AddTree are not reflected
        y = fat.eval(X)
        t.split(t["r"], 1, 0.5)
        at.set_base_score(0, 10.0)
        self.assertTrue(np.array_equal(fat.eval(X), y))
        self.assertEqual(fat.eval_node(X).tolist(), expected)
        self.assertFalse(np.array_equal(FastEvalAddTree(at).eval(X), y))

        # NaNs are rejected unless the caller opts out of the check
        X[1, 1] = np.nan
        self.assertRaises(ValueError, fat.eval, X)
        self.assertRaises(ValueError, fat.eval_node, X)
        self.assertEqual(fat.eval_node(X, check_nan=False).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
