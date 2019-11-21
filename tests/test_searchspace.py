import unittest
from treeck import *
from treeck.plot import TreePlot

def simple():
    at = AddTree.read("tests/models/xgb-covtype-easy.json")
    sp = SearchSpace(at)

    sp.split(1000)

    leafs = sp.leafs()
    print(sp.scores())

    leaf_index_to_plot = 330

    for k, leaf in enumerate(leafs):
        at_pruned = sp.get_pruned_addtree(leaf)
        domains = sp.get_domains(leaf)

        print("{:4}: num_nodes for leaf {:<5}: {} vs {}".format(k, leaf, at.num_nodes(), at_pruned.num_nodes()))

        if k == leaf_index_to_plot:
            for i in range(len(at)):
                p = TreePlot()
                p.add_domains(domains)
                p.add_tree_cmp(at[i], at_pruned[i])
                #p.add_tree(at[i])
                p.add_tree(at_pruned[i])
                p.render("plots/plot-leaf{}-{}".format(k, i))

class TestSearchSpace(unittest.TestCase):
    def test_tree1(self):
        simple()


if __name__ == "__main__":
    simple()
