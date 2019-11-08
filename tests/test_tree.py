import unittest
from treeck import *

class TestTree(unittest.TestCase):

    def test_tree1(self):
        t = Tree()
        t.root().split(LtSplit(1, 1.5))
        t.root().left().set_leaf_value(1.1)
        t.root().right().set_leaf_value(2.2)

        y = t.predict([[1,1,3], [1,2,3]])

        self.assertEqual([1.1,2.2], y)

