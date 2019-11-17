import unittest
from treeck import *

def simple():
    at = AddTree.read("tests/models/xgb-calhouse-easy.json")
    sp = SearchSpace(at)

    sp.split()


class TestSearchSpace(unittest.TestCase):
    def test_tree1(self):
        simple()


if __name__ == "__main__":
    print("hello world!")
    simple()
