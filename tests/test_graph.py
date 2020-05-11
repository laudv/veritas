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

        opt = Optimizer(at, at, {1}, True); # share feature 1 between two trees

        self.assertEqual(opt.xvar_name(0, 0), "x0_0")
        self.assertEqual(opt.xvar_name(0, 1), "x0_1")
        self.assertEqual(opt.xvar_name(1, 0), "x1_0")
        self.assertEqual(opt.xvar_name(1, 1), "x0_1") # shared
        self.assertEqual(opt.xvar_name(0, 2), "x0_2")
        self.assertEqual(opt.xvar_name(1, 2), "x1_2")

        self.assertEqual(opt.num_vertices(0, 0), 4)
        self.assertEqual(opt.num_vertices(0, 1), 5)
        self.assertEqual(opt.num_vertices(1, 0), 4)
        self.assertEqual(opt.num_vertices(1, 1), 5)

        opt.set_smt_program(f"""
(assert (< {opt.xvar_name(0, 0)} 0.0))
(assert (> {opt.xvar_name(0, 1)} 1.2))
(assert (> {opt.xvar_name(1, 0)} 1.0))""")
        opt.prune()

        self.assertEqual(opt.num_vertices(0, 0), 1)
        self.assertEqual(opt.num_vertices(0, 1), 1)
        self.assertEqual(opt.num_vertices(1, 0), 3)
        self.assertEqual(opt.num_vertices(1, 1), 4)

        self.assertEqual(opt.num_independent_sets(0), 2)
        self.assertEqual(opt.num_independent_sets(1), 2)

        opt.merge(2)

        self.assertEqual(opt.num_independent_sets(0), 1)
        self.assertEqual(opt.num_independent_sets(1), 1)

        print(opt)

        #opt.step(1.1);
        #opt.step(1.1, 1.2);

        print(opt.optimize(100, 0.0))
        opt.solutions()

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
