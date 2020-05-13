import unittest, json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO

from treeck import *

def plot_img_solutions(imghat, solutions):
    fig, ax = plt.subplots()
    im = ax.imshow(imghat)
    fig.colorbar(im, ax=ax)
    for i, j, c in [(1,0,'r'), (3,2,"#ffb700"), (3,0,'m'), (1,2,'#ffb700')]:
        hi = False;
        for out0, out1, dom in solutions:
            if i not in dom or j not in dom: continue
            hi = True
            x0, y0 = max(0.0, dom[i].lo), max(0.0, dom[j].lo)
            x1, y1 = min(100.0, dom[i].hi), min(100.0, dom[j].hi)
            w, h = x1-x0, y1-y0
            print((x0, y0), (x1, y1), w, h, out0, out1)
            rect = patches.Rectangle((x0-0.5,y0-0.5),w,h,linewidth=1,edgecolor=c,facecolor='none')
            ax.add_patch(rect)

        if hi and i==3 and j==2: break # no shared variables
    plt.show()

class TestGraph(unittest.TestCase):
    def test_single_tree(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        opt = Optimizer(at, maximize=True)
        notdone = opt.step(100, 3.5)
        self.assertFalse(notdone)
        self.assertEqual(opt.num_solutions(), 2)
        solutions = opt.solutions()
        self.assertEqual(solutions[0][1], 8.0)
        self.assertEqual(solutions[1][1], 4.0)

    def test_two_trees(self):
        at = AddTree()
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

        self.assertEqual(opt.xvar(0, 0), "x0_0")
        self.assertEqual(opt.xvar(0, 1), "x0_1")
        self.assertEqual(opt.xvar(1, 0), "x1_0")
        self.assertEqual(opt.xvar(1, 1), "x0_1") # shared
        self.assertEqual(opt.xvar(0, 2), "x0_2")
        self.assertEqual(opt.xvar(1, 2), "x1_2")

        self.assertEqual(opt.num_vertices(0, 0), 4)
        self.assertEqual(opt.num_vertices(0, 1), 5)
        self.assertEqual(opt.num_vertices(1, 0), 4)
        self.assertEqual(opt.num_vertices(1, 1), 5)

        opt.set_smt_program(f"""
(assert (< {opt.xvar(0, 0)} 0.0))
(assert (> {opt.xvar(0, 1)} 1.2))
(assert (> {opt.xvar(1, 0)} 1.0))""")
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

        notdone = opt.step(100, 0.0)
        self.assertFalse(notdone)
        self.assertEqual(opt.num_solutions(), 5)

    def test_img(self):
        with open("tests/models/xgb-img-very-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-very-easy.json")

        m, M = min(ys), max(ys)
        #img = np.array(ys).reshape((100, 100))

        opt = Optimizer(at, minimize=True)
        print(opt)
        not_done = opt.step(100, 250, -250)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_min = opt.solutions()
        print(solutions_min)
        self.assertTrue(all(x[0] <= y[0] for x, y in zip(solutions_min, solutions_min[1:]))) #sorted?
        min_solution = solutions_min[0]
        print(min_solution, m)

        opt = Optimizer(at, maximize=True)
        print(opt)
        not_done = opt.step(100, 250, -250)
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

        #plot_solutions(imghat, solutions_min[:3])
        #plot_solutions(imghat, solutions_max[:3])

    def test_img2(self): # with constraints
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-easy.json")
        
        #fig, (ax0, ax1) = plt.subplots(1, 2)
        #im0 = ax0.imshow(img)
        #im1 = ax1.imshow(imghat)
        #fig.colorbar(im0, ax=ax0)
        #fig.colorbar(im1, ax=ax1)
        #plt.show()

        m, M = min(ys), max(ys)
        #img = np.array(ys).reshape((100, 100))

        opt = Optimizer(at, minimize=False)
        opt.prune()
        opt.set_smt_program(f"""
(assert (>= {opt.xvar(0, 0)} 50))
(assert (< {opt.xvar(0, 0)} 60))
(assert (>= {opt.xvar(0, 1)} 50))
        """)
        
        while opt.num_solutions() < 50:
            opt.step(100, 250, -250)

        solutions = opt.solutions();

        #print(solutions)
        #plot_img_solutions(imghat, solutions);

        # all solutions must overlap with the region defined in smt
        dom0 = RealDomain(50, 60)
        dom1 = RealDomain(50, 101)
        for _, v, dom in solutions:
            self.assertTrue(dom0.overlaps(dom[0]))
            self.assertTrue(dom1.overlaps(dom[1]))

        # the values found must correspond to the values predicted by the model
        for _, v, dom in solutions:
            x, y = int(max(0.0, dom[1].lo)), int(max(0.0, dom[0].lo)) # right on the edge
            self.assertEqual(v, imghat[y, x])

        self.assertEqual(solutions[0][1], imghat[48, 80])

    def test_img3(self): # with two instances
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-easy.json")

        #opt = Optimizer(at, minimize=False)
        opt = Optimizer(at, at, set(), True) # share no attributes
        opt.set_smt_program(f"""
(assert (= {opt.xvar(0, 0)} {opt.xvar(1, 0)}))
(declare-const h Real)
(assert (= h (- {opt.xvar(0, 1)} {opt.xvar(1, 1)})))
(assert (ite (< h 0) (> h -10) (< h 10)))
""")
#        opt.set_smt_program(f"""
#(assert (> {opt.xvar(0, 0)} 80))
#(assert (> {opt.xvar(0, 1)} 20))
#(assert (< {opt.xvar(0, 1)} 50))
#""")

        current_bounds = [opt.current_bounds()]

        while opt.num_solutions() == 0:
            if not opt.step(25, 250, -250):
                print("no solution")
                break
            current_bounds.append(opt.current_bounds())

        print("previous bounds:", current_bounds)
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in current_bounds], label="lower")
        ax.plot([x[1] for x in current_bounds], label="upper")
        ax.plot([x[1] - x[0] for x in current_bounds], label="diff")
        ax.legend()
        plt.show()

        solutions = opt.solutions();
        print(solutions)
        plot_img_solutions(imghat, solutions[0:1]);

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-easy.json")
        opt = Optimizer(at, at, {2}, False) # feature two not shared
        
        while opt.num_solutions() == 0:
            if not opt.step(100, 0.0):
                break

        print(opt.solutions())

    def test_mnist(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis0-easy.json")
        with open("tests/models/mnist-instances.json") as f:
            instance_key = 0
            instance = np.array(json.load(f)[str(instance_key)])
            vreal = at.predict_single(instance)
            print("predicted value:", vreal)
            #plt.imshow(instance.reshape((28, 28)), cmap="binary")
            #plt.show()

        
        opt = Optimizer(at, minimize=True)

        d = 10
        feat_ids = opt.get_used_feat_ids()[0];
        smt = StringIO()
        for feat_id in feat_ids:
            x = opt.xvar(0, feat_id)
            v = instance[feat_id]
            print(f"(assert (<= {x} {v+d}))", file=smt)
            print(f"(assert (> {x} {v-d}))", file=smt)
        opt.set_smt_program(smt.getvalue())

        bounds_before = opt.propagate_outputs(0)
        num_vertices_before_prune = opt.num_vertices(0)
        opt.prune()
        bounds_after = opt.propagate_outputs(0)
        num_vertices_after_prune = opt.num_vertices(0)
        print(f"prune: num_vertices {num_vertices_before_prune} -> {num_vertices_after_prune}")
        print(f"       bounds {bounds_before} -> {bounds_after}")

        current_bounds = [opt.current_bounds()]
        while opt.num_solutions() == 0:
            if not opt.step(25, 0.0):
                print("no solution")
                break
            current_bounds.append(opt.current_bounds())

        solutions = opt.solutions()
        print([x[0] for x in current_bounds])
        print(solutions)
        instance1 = get_closest_instance(instance, solutions[0][2])
        vfake = at.predict_single(instance1)

        print("predictions:", vreal, vfake, "(", solutions[0][0], ")")

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(instance.reshape((28, 28)), cmap="binary")
        ax1.imshow(instance1.reshape((28, 28)), cmap="binary")
        plt.show()


if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
