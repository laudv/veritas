import unittest, json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def test_img(self):
        with open("tests/models/xgb-img-very-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-very-easy.json")
        
        #print(at)

        #fig, (ax0, ax1) = plt.subplots(1, 2)
        #im0 = ax0.imshow(img)
        #im1 = ax1.imshow(imghat)
        #fig.colorbar(im0, ax=ax0)
        #fig.colorbar(im1, ax=ax1)
        #plt.show()

        m, M = min(ys), max(ys)
        #img = np.array(ys).reshape((100, 100))

        opt = Optimizer(at, minimize=True)
        print(opt)
        not_done = opt.optimize(100, 250, -250)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_min = opt.solutions()
        print(solutions_min)
        self.assertTrue(all(x[0] <= y[0] for x, y in zip(solutions_min, solutions_min[1:]))) #sorted?
        min_solution = solutions_min[0]
        print(min_solution, m)

        opt = Optimizer(at, maximize=True)
        print(opt)
        not_done = opt.optimize(100, 250, -250)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_max = opt.solutions()
        print(solutions_max)
        print(list(x[1] for x in solutions_max))
        self.assertTrue(all(x[1] >= y[1] for x, y in zip(solutions_max, solutions_max[1:]))) #sorted?
        max_solution = solutions_max[0]
        print(max_solution, M)

        # no matter the order in which you generate all solutions, they have to be the same
        for x, y in zip(solutions_min, reversed(solutions_max)):
            self.assertEqual(x[0], y[1])
            self.assertEqual(x[2], y[2])

        # the values found must correspond to the values predicted by the model
        for v, _, dom in solutions_min:
            x, y = int(max(0.0, dom[1].lo)), int(max(0.0, dom[0].lo)) # right on the edge
            self.assertEqual(v, imghat[y, x])

        def plot_solutions(imghat, solutions):
            fig, ax = plt.subplots()
            im = ax.imshow(imghat)
            fig.colorbar(im, ax=ax)
            for out0, out1, dom in solutions:
                x0, y0 = max(0.0, dom[1].lo), max(0.0, dom[0].lo)
                x1, y1 = min(100.0, dom[1].hi), min(100.0, dom[0].hi)
                w, h = x1-x0, y1-y0
                print((x0, y0), (x1, y1), w, h, out0, out1)
                rect = patches.Rectangle((x0-0.5,y0-0.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            plt.show()

        #plot_solutions(imghat, solutions_min[:3])
        #plot_solutions(imghat, solutions_max[:3])


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
