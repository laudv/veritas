import unittest, time
import z3

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

def test_z3():
    at = AddTree.read("tests/models/xgb-covtype-easy.json")
    sp = SearchSpace(at)

    sp.split(10)
    threshold = -7.3

    results = []

    for i, leaf in enumerate(sp.leafs()):
        addtree = sp.get_pruned_addtree(leaf)
        domains = sp.get_domains(leaf)

        #print(addtree)
        print(f"Domains for leaf {i}({leaf}):",
                list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        solver = Z3Solver(sp.num_features(), domains, addtree)
        constraints = [(solver.xvar(35) > 0.5)]
        results.append(solver.verify(constraints, threshold, op=LESS_THAN))

    print(results)


class TestSearchSpace(unittest.TestCase):
    def test_tree1(self):
        simple()


if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    #simple()
    test_z3()
