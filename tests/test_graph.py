import unittest, json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from treeck import *

def plot_img_solutions(imghat, solutions):
    fig, ax = plt.subplots()
    im = ax.imshow(imghat)
    fig.colorbar(im, ax=ax)
    i, j, c = 1, 0, "r"
    for out0, out1, dom in solutions:
        hi = True
        x0, y0 = max(0.0, dom[i].lo), max(0.0, dom[j].lo)
        x1, y1 = min(100.0, dom[i].hi), min(100.0, dom[j].hi)
        w, h = x1-x0, y1-y0
        print((x0, y0), (x1, y1), w, h, out0, out1)
        rect = patches.Rectangle((x0-0.5,y0-0.5),w,h,lw=1,ec=c,fc='none')
        ax.add_patch(rect)

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

        opt = Optimizer(maximize=at)
        notdone = opt.step(100, min_output=3.5)
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

        opt = Optimizer(minimize=at, maximize=at, matches={1}, match_is_reuse=True); # share feature 1 between two trees
        opt.enable_smt()

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

        notdone = opt.step(100)
        self.assertFalse(notdone)
        self.assertEqual(opt.num_solutions(), 5)

        for fid in opt.get_used_feat_ids()[0]:
            print(0, fid, "->", opt.xvar_id(0, fid))
        for fid in opt.get_used_feat_ids()[1]:
            print(1, fid, "->", opt.xvar_id(1, fid))
        print()
        
        for s0, s1, sd in opt.solutions():
            print(s0, s1, sd)

        opt = Optimizer(minimize=at)
        self.assertEqual(opt.num_vertices(0, 0), 4)
        self.assertEqual(opt.num_vertices(0, 1), 5)
        opt.prune([1.2, 10.0, True], 0.5)
        self.assertEqual(opt.num_vertices(0, 0), 2)
        self.assertEqual(opt.num_vertices(0, 1), 1)
        

    def test_img(self):
        with open("tests/models/xgb-img-very-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-very-easy.json")

        m, M = min(ys), max(ys)
        #img = np.array(ys).reshape((100, 100))

        opt = Optimizer(minimize=at)
        print(opt)
        not_done = opt.step(100)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_min = opt.solutions()
        print("solutions_min", list(x[0] for x in solutions_min))
        self.assertTrue(all(x[0] <= y[0] for x, y in zip(solutions_min, solutions_min[1:]))) #sorted?
        min_solution = solutions_min[0]
        print(min_solution, m)

        opt = Optimizer(maximize=at)
        print(opt)
        not_done = opt.step(100)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_max = opt.solutions()
        print("solutions_max", list(x[1] for x in solutions_max))
        self.assertTrue(all(x[1] >= y[1] for x, y in zip(solutions_max, solutions_max[1:]))) #sorted?
        max_solution = solutions_max[0]
        print("max_solution:", max_solution, M)

        # no matter the order in which you generate all solutions, they have to be the same
        for x, y, real in zip(solutions_min, reversed(solutions_max), sorted(list(set(ys)))):
            self.assertEqual(x[0], y[1]) # outputs (min, max)
            self.assertEqual(real, y[1])
            self.assertEqual(x[2], y[2]) # domains

        # the values found must correspond to the values predicted by the model
        for v, _, dom in solutions_min:
            x, y = int(max(0.0, dom[1].lo)), int(max(0.0, dom[0].lo)) # right on the edge
            self.assertEqual(v, imghat[y, x])

        plot_img_solutions(imghat, solutions_min[:3])
        plot_img_solutions(imghat, solutions_max[:3])

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

        opt = Optimizer(maximize=at)
        opt.enable_smt()
        opt.set_smt_program(f"""
(assert (>= {opt.xvar(1, 0)} 50))
(assert (< {opt.xvar(1, 0)} 60))
(assert (>= {opt.xvar(1, 1)} 50))
        """)
        opt.prune()
        
        while opt.num_solutions() < 50:
            opt.step(100)

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

        opt = Optimizer(maximize=at)
        opt.enable_smt()
        #opt = Optimizer(at, at, set(), True) # share no attributes
#        opt.set_smt_program(f"""
#(assert (= {opt.xvar(0, 0)} {opt.xvar(1, 0)}))
#(declare-const h Real)
#(assert (= h (- {opt.xvar(0, 1)} {opt.xvar(1, 1)})))
#(assert (ite (< h 0) (> h -10) (< h 10)))
#""")
        opt.set_smt_program(f"""
(assert (> {opt.xvar(1, 0)} 90))
(assert (> {opt.xvar(1, 1)} 50))
(assert (< {opt.xvar(1, 1)} 70))
""")

        current_bounds = [opt.current_bounds()]

        while opt.num_solutions() == 0:
            if not opt.step(1):
                print("no solution")
                break
            current_bounds.append(opt.current_bounds())

        print("previous bounds:", current_bounds)
        fig, ax = plt.subplots()
        #ax.plot([x[0] for x in current_bounds], label="lower")
        ax.plot([x[1] for x in current_bounds], label="upper")
        #ax.plot([x[1] - x[0] for x in current_bounds], label="diff")
        ax.legend()
        plt.show()

        solutions = opt.solutions();
        print(solutions)
        plot_img_solutions(imghat, solutions[0:1]);

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-easy.json")
        opt = Optimizer(minimize=at, maximize=at, matches={2}, match_is_reuse=False) # feature two not shared
        
        while opt.num_solutions() == 0:
            if not opt.step(100, min_output_difference=0.0):
                break

        print(opt.solutions())

    def test_mnist_prune(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis1-easy.json")
        with open("tests/models/mnist-instances.json") as f:
            example_key = 1
            example = np.array(json.load(f)[str(example_key)])
            vreal = at.predict_single(example)
            print("predicted value:", vreal)

        d = 1.0001

        opt_smt = Optimizer(minimize=at)
        opt_box = Optimizer(minimize=at)

        xvar_id_map = get_xvar_id_map(opt_smt, instance=0)
        self.assertEqual(xvar_id_map, get_xvar_id_map(opt_box, instance=0))

        self.assertEqual(opt_smt.current_bounds(), opt_box.current_bounds())
        self.assertEqual(opt_smt.num_vertices(0), opt_box.num_vertices(0))

        opt_smt.enable_smt()
        opt_smt.set_smt_program(get_example_box_smt(opt_smt, 0, example, d))
        opt_smt.prune()
        
        opt_box.prune(list(example), d)

        #print(opt_smt)
        #print(opt_box)
        #print({k : example[fid] for k, fid in xvar_id_map.items()})

        self.assertEqual(opt_smt.current_bounds(), opt_box.current_bounds())
        self.assertEqual(opt_smt.num_vertices(0), opt_box.num_vertices(0))

    def test_mnist(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis1-easy.json")
        with open("tests/models/mnist-instances.json") as f:
            example_key = 1
            example = np.array(json.load(f)[str(example_key)])
            vreal = at.predict_single(example)
            print("predicted value:", vreal)
            #plt.imshow(example.reshape((28, 28)), cmap="binary")
            #plt.show()

        
        opt = Optimizer(minimize=at)
        opt.enable_smt()

        d = 1.1
        opt.set_smt_program(get_example_box_smt(opt, 0, example, d))

        bounds_before = opt.current_bounds()[0]
        num_vertices_before_prune = opt.num_vertices(0)
        opt.prune()
        bounds_after = opt.current_bounds()[0]
        num_vertices_after_prune = opt.num_vertices(0)
        print(f"prune: num_vertices {num_vertices_before_prune} -> {num_vertices_after_prune}")
        print(f"       bounds {bounds_before} -> {bounds_after}")

        current_bounds = [opt.current_bounds()]
        while opt.num_solutions() == 0:
            if not opt.step(25):
                print("no solution")
                break
            current_bounds.append(opt.current_bounds())

        self.assertTrue(opt.num_solutions() > 0)

        solutions = opt.solutions()
        print([x[0] for x in current_bounds])
        #print("solutions", solutions)
        xvar_id_map = get_xvar_id_map(opt, instance=0)
        print("xvar_id_map", xvar_id_map)
        example1 = get_closest_example(xvar_id_map, example, solutions[0][2])
        vfake = at.predict_single(example1)
        diff = min(example1 - example), max(example1 - example)
        print("diff", diff)

        print("predictions:", vreal, vfake, "(", solutions[0][0], ")")

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(example.reshape((28, 28)), cmap="binary")
        ax0.set_title(f"{vreal:.4f}")
        ax1.imshow(example1.reshape((28, 28)), cmap="binary")
        ax1.set_title(f"{vfake:.4f}")
        im = ax2.imshow((example1-example).reshape((28, 28)), cmap="binary")
        fig.colorbar(im, ax=ax2)
        plt.show()

    def test_mnist2(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis0-easy.json")
        with open("tests/models/mnist-instances.json") as f:
            example_key = 0
            example = np.array(json.load(f)[str(example_key)])
            vreal = at.predict_single(example)
            print("predicted value:", vreal)
            #plt.imshow(example.reshape((28, 28)), cmap="binary")
            #plt.show()

        opt = Optimizer(minimize=at)

        d = 1.1

        bounds_before = opt.current_bounds()[0]
        num_vertices_before_prune = opt.num_vertices(0)
        opt.prune(list(example), d)
        bounds_after = opt.current_bounds()[0]
        num_vertices_after_prune = opt.num_vertices(0)
        print(f"prune: num_vertices {num_vertices_before_prune} -> {num_vertices_after_prune}")
        print(f"       bounds {bounds_before} -> {bounds_after}")

        current_bounds = [opt.current_bounds()]
        while opt.num_solutions() == 0:
            if not opt.step(25):
                print("no solution")
                break
            current_bounds.append(opt.current_bounds())

        self.assertTrue(opt.num_solutions() > 0)

        solutions = opt.solutions()
        print([x[0] for x in current_bounds])
        #print("solutions", solutions)
        xvar_id_map = get_xvar_id_map(opt, instance=0)
        print("xvar_id_map", xvar_id_map)
        example1 = get_closest_example(xvar_id_map, example, solutions[0][2])
        vfake = at.predict_single(example1)
        diff = min(example1 - example), max(example1 - example)
        print("diff", diff)

        print("predictions:", vreal, vfake, "(", solutions[0][0], ")")

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(example.reshape((28, 28)), cmap="binary")
        ax0.set_title(f"{vreal:.4f}")
        ax1.imshow(example1.reshape((28, 28)), cmap="binary")
        ax1.set_title(f"{vfake:.4f}")
        im = ax2.imshow((example1-example).reshape((28, 28)), cmap="binary")
        fig.colorbar(im, ax=ax2)
        plt.show()

    def test_simplify(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 8.0)
        t.set_leaf_value(t.right( t.left(t.root())), 4.0)
        t.set_leaf_value( t.left(t.right(t.root())), 2.0)
        t.set_leaf_value(t.right(t.right(t.root())), 1.0)

        opt = Optimizer(maximize=at, simplify=(1, 2.0))
        self.assertEqual(opt.num_vertices(1), 3)

    def test_simplify2(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 5)
        t.split(t.right(t.root()), 1, 5)
        t.split(t.left(t.right(t.root())), 2, 5)
        t.set_leaf_value(t.left(t.root()), 1)
        t.set_leaf_value( t.left(t.left(t.right(t.root()))), 2)
        t.set_leaf_value(t.right(t.left(t.right(t.root()))), 3)
        t.set_leaf_value(t.right(t.right(t.root())), 4)

        #print(at)

        opt = Optimizer(minimize=at, simplify=(0, 2.0))
        self.assertEqual(opt.num_vertices(0), 2)
        self.assertFalse(opt.step(10))
        self.assertEqual(opt.solutions()[0][0], 1)
        self.assertEqual(opt.solutions()[1][0], 4)

    def test_simplify3(self):
        with open("tests/models/xgb-img-very-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        at = AddTree.read(f"tests/models/xgb-img-very-easy.json")
        print(at)

        opt0 = Optimizer(minimize=at)
        opt1 = Optimizer(minimize=at, simplify=(0, 50.0))

        opt0.step(100)
        opt1.step(100)

        print(opt0.solutions()[0])
        print(opt1.solutions()[0])

        self.assertTrue(opt1.solutions()[0][0] - opt0.solutions()[0][0] < 50.0)

        #plot_img_solutions(imghat, opt0.solutions()[0:1])
        #plot_img_solutions(imghat, opt1.solutions()[0:1])

    def test_simplify4(self):
        at = AddTree.read(f"tests/models/xgb-calhouse-intermediate.json")

        opt0 = Optimizer(minimize=at)
        opt1 = Optimizer(minimize=at, simplify=(0, 2.0))

        print(opt0.num_vertices(0))
        print(opt1.num_vertices(0))

        #opt0.set_ara_eps(0.5, 0.1)
        while opt0.num_solutions() == 0:
            opt0.step(100)
        #opt1.set_ara_eps(0.5, 0.1)
        while opt1.num_solutions() == 0:
            opt1.step(100)

        print(opt0.solutions())
        print(opt1.solutions())

        print(opt0.nsteps())
        print(opt1.nsteps())


if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
