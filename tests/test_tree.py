import unittest
from treeck import *

class TestTree(unittest.TestCase):

    def test_tree1(self):
        at = AddTree()
        t = at.add_tree()
        t.root().split(LtSplit(1, 1.5))
        t.root().left().set_leaf_value(1.1)
        t.root().right().set_leaf_value(2.2)

        y = t.predict([[1,1,3], [1,2,3]])

        self.assertEqual([1.1,2.2], y)

    def test_tree_json(self):
        at = AddTree()
        t = at.add_tree()
        t.root().split(LtSplit(1, 1.5))
        t.root().left().split(LtSplit(2, 0.12))
        t.root().left().left().set_leaf_value(0.25)
        t.root().left().right().set_leaf_value(0.45)
        t.root().right().set_leaf_value(2.2)

        s = at.to_json();
        att = AddTree.from_json(s)
        tt = att[0]

        self.assertTrue(tt[0].is_internal())
        self.assertTrue(tt[1].is_internal())
        self.assertEquals(tt[0].get_split().feat_id, 1)
        self.assertEquals(tt[1].get_split().feat_id, 2)
        self.assertEquals(tt[0].get_split().split_value, 1.5)
        self.assertEquals(tt[1].get_split().split_value, 0.12)
        self.assertTrue(tt[2].is_leaf())
        self.assertTrue(tt[3].is_leaf())
        self.assertTrue(tt[4].is_leaf())
        self.assertEquals(tt[2].leaf_value(), 2.2)
        self.assertEquals(tt[3].leaf_value(), 0.25)
        self.assertEquals(tt[4].leaf_value(), 0.45)

