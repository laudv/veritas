import unittest

from treeck import *

class TestGraph(unittest.TestCase):
    def test_single_tree(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        graph = KPartiteGraph(at)

    def test_two_trees(self):
        at = AddTree()
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        t = at.add_tree();
        t.split(t.root(), 0, 3)
        t.split( t.left(t.root()), 1, 1.2)
        t.split(t.right(t.root()), 1, 3.3)
        t.split(t.right(t.right(t.root())), 2)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value( t.left(t.right(t.right(t.root()))), 0.5)
        t.set_leaf_value(t.right(t.right(t.right(t.root()))), 0.6)

        print(at)
        graph = KPartiteGraph(at)
        graph.propagate_outputs()
        graph.prune("""
(declare-fun x () Real)
(assert (> x  0.0))
        """)
        print(graph)

        #print(graph)
        #print("outputs: ", graph.propagate_outputs())
        #graph.merge(2);
        #print("outputs: ", graph.propagate_outputs())
        #print(graph)

        #print("\n== MAX ======================")
        #find = MaxKPartiteGraphFind(graph)
        #print("done?", not find.steps(100))
        #max_solutions = find.solutions()
        #print(len(max_solutions))
 
        #print("\n== MIN ======================")
        #find = MinKPartiteGraphFind(graph)
        #print("done?", not find.steps(100))
        #min_solutions = find.solutions()
        #print(len(min_solutions), graph.nsteps, graph.nupdate_fails)

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-easy.json")

        graph = KPartiteGraph(at)

        print(graph)

        print("outputs: ", graph.propagate_outputs(), ", size", len(graph), ", #vertex", graph.num_vertices())
        #graph.merge(2);
        #print("outputs: ", graph.propagate_outputs(), ", size", len(graph), ", #vertex", graph.num_vertices())
        #graph.merge(2);
        #print(graph)
        #print("outputs: ", graph.propagate_outputs(), ", size", len(graph), ", #vertex", graph.num_vertices())
        #graph.merge(2);
        #print("outputs: ", graph.propagate_outputs(), ", size", len(graph), ", #vertex", graph.num_vertices())
        #graph.merge(2);
        #print("outputs: ", graph.propagate_outputs(), ", size", len(graph), ", #vertex", graph.num_vertices())

        print("\n== MAX ======================")
        find = MaxKPartiteGraphFind(graph)
        print("done?", not find.steps(1000))
        max_solutions = find.solutions()
        print("#sol", len(max_solutions),
              "#steps", find.nsteps,
              "#nfails", find.nupdate_fails)
 
        #print("\n== MIN ======================")
        #find = MinKPartiteGraphFind(graph)
        #print("done?", not find.steps(100))
        #min_solutions = find.solutions()
        #print(len(min_solutions))

if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
