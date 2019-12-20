import unittest, pickle
from treeck import *

class TestTree(unittest.TestCase):

    def test_tree1(self):
        at = AddTree(2)
        t = at.add_tree()
        t.split(t.root(), 1, 1.5)
        t.set_leaf_value(t.left(t.root()), 1.1)
        t.set_leaf_value(t.right(t.root()), 2.2)

        y = t.predict([[1,1,3], [1,2,3]])

        self.assertEqual([1.1,2.2], y)

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
        self.assertEqual(tt.get_split(0), (1, 1.5))
        self.assertEqual(tt.get_split(1), (2, 0.12))
        self.assertTrue(tt.is_leaf(2))
        self.assertTrue(tt.is_leaf(3))
        self.assertTrue(tt.is_leaf(4))
        self.assertEqual(tt.get_leaf_value(2), 2.2)
        self.assertEqual(tt.get_leaf_value(3), 0.25)
        self.assertEqual(tt.get_leaf_value(4), 0.45)

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

        self.assertEqual(s[1], [1.5, 2.0])
        self.assertEqual(s[2], [0.12])
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
        self.assertEqual(t.get_leaf_value(t.left(t.root())), 0.45)
        self.assertEqual(t.get_leaf_value(t.right(t.root())), 2.2)

    def test_pickle(self):
        at = AddTree.read("tests/models/xgb-covtype-easy.json")
        att = pickle.loads(pickle.dumps(at))
        self.assertEqual(at.num_features(), att.num_features())
        self.assertEqual(at.num_nodes(), att.num_nodes())

if __name__ == "__main__":
    unittest.main()
