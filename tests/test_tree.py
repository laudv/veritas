import unittest, pickle
from treeck import *

class TestTree(unittest.TestCase):

    def myAssertAlmostEqual(self, a, b, eps=1e-6):
        self.assertTrue(type(a) == type(b))
        if isinstance(a, list) or isinstance(a, tuple):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.myAssertAlmostEqual(x, y, eps=eps)
        elif isinstance(a, float):
            self.assertAlmostEqual(a, b, delta=eps)
        else:
            self.assertEqual(a, b)

    def test_tree1(self):
        at = AddTree(2)
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.set_leaf_value(t.left(t.root()), 1.1)
        t.set_leaf_value(t.right(t.root()), 2.2)

        y = t.predict([[1,1,3], [1,2,3]])

        self.myAssertAlmostEqual([1.1, 2.2], y)

    def test_tree_json(self):
        at = AddTree(3)
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)

        s = at.to_json();
        att = AddTree.from_json(s)
        tt = att[0]

        self.assertTrue(tt.is_internal(0))
        self.assertTrue(tt.is_internal(1))
        self.myAssertAlmostEqual(tt.get_split(0), ("lt", 1, 1.5))
        self.myAssertAlmostEqual(tt.get_split(1), ("lt", 2, 0.12))
        self.assertTrue(tt.is_leaf(2))
        self.assertTrue(tt.is_leaf(3))
        self.assertTrue(tt.is_leaf(4))
        self.assertAlmostEqual(tt.get_leaf_value(2), 2.2)
        self.assertAlmostEqual(tt.get_leaf_value(3), 0.25)
        self.assertAlmostEqual(tt.get_leaf_value(4), 0.45)

    def test_boolsplit(self):
        at = AddTree(3)
        t = at.add_tree()
        t.split(t.root(), 0, 1.5)
        t.split(t.left(t.root()), 1, 1.0)
        t.split(t.right(t.root()), 2)
        t.set_leaf_value(t.left(t.left(t.root())), 1.0)
        t.set_leaf_value(t.right(t.left(t.root())), 2.0)
        t.set_leaf_value(t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        self.assertEqual(t.get_split( t.left(t.root())), ("lt", 1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), ("bool", 2))

        self.assertEqual([1.0, 2.0, 4.0, 8.0], at.predict([
            [0.0, 0.5, True], [0.0, 1.5, True],
            [2.0, 0.5, False], [2.0, 0.5, True]]))

        s = at.to_json();
        att = AddTree.from_json(s)
        t = att[0]

        self.assertEqual(t.get_split( t.left(t.root())), ("lt", 1, 1.0))
        self.assertEqual(t.get_split(t.right(t.root())), ("bool", 2))

    def test_addtree_get_splits(self):
        at = AddTree(3)
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.split(t.left(t.root()), 2, 0.12)
        t.set_leaf_value(t.left(t.left(t.root())), 0.25)
        t.set_leaf_value(t.right(t.left(t.root())), 0.45)
        t.set_leaf_value(t.right(t.root()), 2.2)

        t = at.add_tree()
        t.split(t.root(), 1, 2.0)
        t.set_leaf_value(t.left(t.root()), 0.5)
        t.set_leaf_value(t.right(t.root()), 2.3)

        s = at.get_splits()

        self.myAssertAlmostEqual(s[1], [1.5, 2.0])
        self.myAssertAlmostEqual(s[2], [0.12])
        self.assertEqual(sorted(list(s.keys())), [1, 2])

    def test_skip_branch_left(self):
        at = AddTree(2)
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
        at = AddTree(3)
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
        at = AddTree(3)
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
        self.assertEqual(at.num_features(), att.num_features())
        self.assertEqual(at.num_nodes(), att.num_nodes())

if __name__ == "__main__":
    unittest.main()
