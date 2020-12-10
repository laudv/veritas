import unittest, json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timeit

from veritas import *

from experiments.util import get_ara_bound, filter_solutions

def plot_img_solutions(imghat, solutions):
    fig, ax = plt.subplots()
    im = ax.imshow(imghat)
    fig.colorbar(im, ax=ax)
    i, j, c = 1, 0, "r"
    for s in solutions:
        dom = s.box()
        hi = True
        x0, y0 = max(0.0, dom[i].lo), max(0.0, dom[j].lo)
        x1, y1 = min(100.0, dom[i].hi), min(100.0, dom[j].hi)
        w, h = x1-x0, y1-y0
        print((x0, y0), (x1, y1), w, h, s.output0, s.output1)
        rect = patches.Rectangle((x0-0.5,y0-0.5),w,h,lw=1,ec=c,fc='none')
        ax.add_patch(rect)

    plt.show()

class TestGraph(unittest.TestCase):
    def test_single_tree(self):
        dummy = AddTree()
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        finfo = FeatInfo(dummy, at, set(), False);
        print(finfo.feat_ids0())
        print(finfo.feat_ids1())
        g0 = KPartiteGraph(dummy, finfo, 0)
        print(g0)
        g1 = KPartiteGraph(at, finfo, 1)
        print(g1)
        opt = KPartiteGraphOptimize(g0, g1, KPartiteGraphOptimizeHeuristic.RECOMPUTE)
        notdone = opt.steps(100, min_output=3.5)
        self.assertFalse(notdone)
        self.assertEqual(opt.num_solutions(), 2)
        solutions = opt.solutions
        self.assertEqual(solutions[0].output1, 8.0)
        self.assertEqual(solutions[1].output1, 4.0)
        print(solutions[0].box())

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
        s = SMTSolver(opt.feat_info, opt.at0, opt.at1)

        self.assertEqual(s.xvar_name(0, 0), "x0_0")
        self.assertEqual(s.xvar_name(0, 1), "x0_1")
        self.assertEqual(s.xvar_name(1, 0), "x1_0")
        self.assertEqual(s.xvar_name(1, 1), "x0_1") # shared
        self.assertEqual(s.xvar_name(0, 2), "x0_2")
        self.assertEqual(s.xvar_name(1, 2), "x1_2")

        self.assertEqual(opt.g0.num_vertices_in_set(0), 4)
        self.assertEqual(opt.g0.num_vertices_in_set(1), 5)
        self.assertEqual(opt.g1.num_vertices_in_set(0), 4)
        self.assertEqual(opt.g1.num_vertices_in_set(1), 5)

        opt.prune_smt("""
(assert (< {instance0_var_0} 0.0))
(assert (> {instance0_var_1} 1.2))
(assert (> {g0} 1.0))""", var_prefix0="instance0_var_")

        self.assertEqual(opt.g0.num_vertices_in_set(0), 1)
        self.assertEqual(opt.g0.num_vertices_in_set(1), 1)
        self.assertEqual(opt.g1.num_vertices_in_set(0), 3)
        self.assertEqual(opt.g1.num_vertices_in_set(1), 4)

        self.assertEqual(opt.g0.num_independent_sets(), 2)
        self.assertEqual(opt.g1.num_independent_sets(), 2)

        opt.g0.merge(2)
        opt.g1.merge(2)

        self.assertEqual(opt.g0.num_independent_sets(), 1)
        self.assertEqual(opt.g1.num_independent_sets(), 1)

        notdone = opt.steps(100)
        self.assertFalse(notdone)
        self.assertEqual(opt.num_solutions(), 5)

        for fid in opt.feat_info.feat_ids0():
            print(0, fid, "->", s.xvar_id(0, fid))
        for fid in opt.feat_info.feat_ids1():
            print(1, fid, "->", s.xvar_id(1, fid))
        print()
        
        for s in opt.solutions():
            print(s.output0, s.output1, s.box())

        opt = Optimizer(minimize=at)
        self.assertEqual(opt.g0.num_vertices_in_set(0), 4)
        self.assertEqual(opt.g0.num_vertices_in_set(1), 5)
        opt.prune_example([1.2, 10.0, True], 0.5)
        self.assertEqual(opt.g0.num_vertices_in_set(0), 2)
        self.assertEqual(opt.g0.num_vertices_in_set(1), 1)

    def test_neg_leafs(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 1)
        t.set_leaf_value( t.left(t.root()), 1)
        t.set_leaf_value(t.right(t.root()), 1)
        t = at.add_tree();
        t.split(t.root(), 0)
        t.set_leaf_value( t.left(t.root()), 1)
        t.set_leaf_value(t.right(t.root()), 1)
        t = at.add_tree();
        t.split(t.root(), 0)
        t.set_leaf_value( t.left(t.root()), -10)
        t.set_leaf_value(t.right(t.root()), -20)
        t = at.add_tree()
        t.split(t.root(), 2)
        t.set_leaf_value( t.left(t.root()), 1)
        t.set_leaf_value(t.right(t.root()), 1)

        opt = Optimizer(maximize=at)
        opt.steps(100) # should not fail because "assertion error: bound increase for same eps"
        
    def test_three_trees_dynprog(self):
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
        t.set_leaf_value( t.left( t.left(t.root())), 1)
        t.set_leaf_value(t.right( t.left(t.root())), 2)
        t.set_leaf_value( t.left(t.right(t.root())), 3)
        t.set_leaf_value( t.left(t.right(t.right(t.root()))), 5)
        t.set_leaf_value(t.right(t.right(t.right(t.root()))), 6)

        t = at.add_tree();
        t.split(t.root(), 0, 1.2)
        t.split( t.left(t.root()), 0, 0.5)
        t.split(t.right(t.root()), 0, 3.9)
        t.set_leaf_value( t.left( t.left(t.root())), 100)
        t.set_leaf_value(t.right( t.left(t.root())), 200)
        t.set_leaf_value( t.left(t.right(t.root())), 300)
        t.set_leaf_value(t.right(t.right(t.root())), 400)
        
        # share feature 1 between two trees
        opt0 = Optimizer(minimize=at, maximize=at, matches={1}, match_is_reuse=True, use_dyn_prog_heuristic=True);
        opt0.steps(1000)
        opt1 = Optimizer(minimize=at, maximize=at, matches={1}, match_is_reuse=True); # share feature 1 between two trees
        opt1.steps(1000)

        print("num_steps DYN_PROG ", sum(opt0.num_steps()))
        print("num_steps RECOMPUTE", sum(opt1.num_steps()))

        self.assertEqual(opt0.num_solutions(), opt1.num_solutions()) # check solutions dynprog vs recompute

        sols0 = sorted(opt0.solutions(), key=lambda x: x.output0)
        sols1 = sorted(opt1.solutions(), key=lambda x: x.output0)
        sols0 = sorted(sols0, key=Solution.output_difference) # assuming stable sort
        sols1 = sorted(sols1, key=Solution.output_difference)

        for s0, s1 in zip(sols0, sols1):
            self.assertEqual(s0.output0, s1.output0)
            self.assertEqual(s0.output1, s1.output1)
            self.assertEqual(s0.box(), s1.box())

    def test_img(self):
        with open("tests/models/xgb-img-very-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-very-easy.json")

        m, M = min(ys), max(ys)
        #img = np.array(ys).reshape((100, 100))

        opt = Optimizer(minimize=at)
        not_done = opt.steps(100)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_min = opt.solutions()
        #print("solutions_min", list(s.output0 for s in solutions_min))
        self.assertTrue(all(x.output0 <= y.output0 for x, y in zip(solutions_min, solutions_min[1:]))) #sorted?
        min_solution = solutions_min[0]
        print(min_solution, m)

        opt = Optimizer(maximize=at)
        not_done = opt.steps(100)
        self.assertFalse(not_done)
        self.assertEqual(opt.num_solutions(), 32);
        solutions_max = opt.solutions()
        #print("solutions_max", list(s.output1 for s in solutions_max))
        self.assertTrue(all(x.output1 >= y.output1 for x, y in zip(solutions_max, solutions_max[1:]))) #sorted?
        max_solution = solutions_max[0]
        print("max_solution:", max_solution, M)

        # no matter the order in which you generate all solutions, they have to be the same
        for x, y, real in zip(solutions_min, reversed(solutions_max), sorted(list(set(ys)))):
            self.assertEqual(x.output0, y.output1) # outputs (min, max)
            self.assertEqual(real, y.output1)
            self.assertEqual(x.box(), y.box()) # domains

        # the values found must correspond to the values predicted by the model
        for s in solutions_min:
            dom = s.box()
            x, y = int(max(0.0, dom[1].lo)), int(max(0.0, dom[0].lo)) # right on the edge
            self.assertEqual(s.output0, imghat[y, x])

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

        opt = Optimizer(maximize=at, use_dyn_prog_heuristic=False)
        opt.prune_smt("""
(assert (>= {g0} 50))
(assert (< {g0} 60))
(assert (>= {g1} 50))
        """)
        
        while opt.num_solutions() < 50:
            opt.steps(100)

        solutions = opt.solutions();

        #print(solutions)
        #plot_img_solutions(imghat, solutions);

        # all solutions must overlap with the region defined in smt
        dom0 = RealDomain(50, 60)
        dom1 = RealDomain(50, 101)
        for s in solutions:
            box = s.box()
            self.assertTrue(dom0.overlaps(box[0]))
            self.assertTrue(dom1.overlaps(box[1]))

        # the values found must correspond to the values predicted by the model
        for s in solutions:
            box = s.box()
            x, y = int(max(0.0, box[1].lo)), int(max(0.0, box[0].lo)) # right on the edge
            self.assertEqual(s.output1, imghat[y, x])

        self.assertEqual(solutions[0].output1, imghat[48, 80])

    def test_img3(self): # with two instances
        with open("tests/models/xgb-img-hard-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-hard.json")

        opt = Optimizer(maximize=at)#, use_dyn_prog_heuristic=True)
        opt.prune_smt("""
(assert (> {g0} 90))
(assert (> {g1} 50))
(assert (< {g1} 70))
""")

        while opt.num_solutions() == 0:
            if not opt.steps(1):
                print("no solution")
                break

        solutions = sorted(opt.solutions(), key=lambda x: x.output1, reverse=True)
        print("solutions: ", solutions)

        self.assertEqual(solutions[0].output1, 122.28324127197266)
        self.assertEqual(solutions[0].box()[0], RealDomain(93, 94))
        self.assertEqual(solutions[0].box()[1], RealDomain(69, 70))

        #print("previous bounds:", current_bounds)
        print("number of steps:", opt.num_steps())
        print("number of solutions:", len(solutions))
        print("best solution:", solutions[0].output1)
        fig, ax = plt.subplots()
        print(opt.bounds)
        ax.plot([x[1] for x in opt.bounds], label="upper")
        ax.legend()
        plt.show()

        solutions = opt.solutions();
        print(solutions)
        plot_img_solutions(imghat, solutions[0:1]);


    def test_img_ara(self): # with two instances
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        imghat = np.array(ys).reshape((100, 100))
        img = imageio.imread("tests/data/img.png")
        at = AddTree.read("tests/models/xgb-img-easy.json")

        opt = Optimizer(maximize=at)#, use_dyn_prog_heuristic=True)

        opt.set_eps(0.1)
        while opt.num_solutions() == 0:
            if not opt.steps(1):
                print("no solution")
                break
        opt.set_eps(0.5)
        while opt.num_solutions() == 1:
            if not opt.steps(1):
                print("no solution")
                break
        opt.set_eps(1.0)
        while opt.num_solutions() == 2:
            if not opt.steps(1):
                print("no solution")
                break

        sols = opt.solutions()
        self.assertTrue(sols[0].output_difference() <= sols[1].output_difference())
        self.assertTrue(sols[1].output_difference() <= sols[2].output_difference())
        plot_img_solutions(imghat, sols);

    def test_calhouse(self):
        at = AddTree.read("tests/models/xgb-calhouse-easy.json")
        opt = Optimizer(minimize=at, maximize=at, matches={2}, match_is_reuse=False, # feature two not shared
                use_dyn_prog_heuristic=False)
        
        while opt.num_solutions() == 0:
            if not opt.steps(100, min_output_difference=0.0):
                break

        print("num_steps", opt.num_steps())
        #for sol in opt.solutions():
        #    print(sol)

        solutions = sorted(opt.solutions(), key=Solution.output_difference, reverse=True)
        self.assertEqual(solutions[0].output0, 3.848149061203003)
        self.assertEqual(solutions[0].output1, 4.45944881439209)

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
        self.assertEqual(opt_smt.g0.num_vertices(), opt_box.g0.num_vertices())

        opt_smt.prune_smt(get_example_box_smt(opt_smt, 0, example, d))
        opt_box.prune_example(list(example), d)

        #print(opt_smt)
        #print(opt_box)
        #print({k : example[fid] for k, fid in xvar_id_map.items()})

        self.assertEqual(opt_smt.current_bounds(), opt_box.current_bounds())
        self.assertEqual(opt_smt.g0.num_vertices(), opt_box.g0.num_vertices())

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
        d = 1.1

        bounds_before = opt.current_bounds()[0]
        num_vertices_before_prune = opt.g0.num_vertices()
        opt.prune_smt(get_example_box_smt(opt, 0, example, d))
        bounds_after = opt.current_bounds()[0]
        num_vertices_after_prune = opt.g0.num_vertices()
        print(f"prune: num_vertices {num_vertices_before_prune} -> {num_vertices_after_prune}")
        print(f"       bounds {bounds_before} -> {bounds_after}")

        #opt.use_dyn_prog_heuristic()
        while opt.num_solutions() == 0:
            if not opt.steps(25):
                print("no solution")
                break

        self.assertTrue(opt.num_solutions() > 0)

        solutions = opt.solutions()
        print([x[0] for x in opt.bounds])
        #print("solutions", solutions)
        xvar_id_map = get_xvar_id_map(opt, instance=0)
        print("xvar_id_map", xvar_id_map)
        example1 = get_closest_example(xvar_id_map, example, solutions[0].box())
        vfake = at.predict_single(example1)
        diff = min(example1 - example), max(example1 - example)
        print("diff", diff)

        print("predictions:", vreal, vfake, "(", solutions[0].output0, ")")
        print("num_steps:", opt.num_steps())
        self.assertTrue(abs(solutions[0].output0 - vfake) < 1e-5)

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
        num_vertices_before_prune = opt.g0.num_vertices()
        opt.prune_example(list(example), d)
        bounds_after = opt.current_bounds()[0]
        num_vertices_after_prune = opt.g0.num_vertices()
        print(f"prune: num_vertices {num_vertices_before_prune} -> {num_vertices_after_prune}")
        print(f"       bounds {bounds_before} -> {bounds_after}")

        #opt.use_dyn_prog_heuristic()
        while opt.num_solutions() == 0:
            if not opt.steps(25):
                print("no solution")
                break

        self.assertTrue(opt.num_solutions() > 0)

        solutions = opt.solutions()
        print([x[0] for x in opt.bounds])
        #print("solutions", solutions)
        xvar_id_map = get_xvar_id_map(opt, instance=0)
        print("xvar_id_map", xvar_id_map)
        example1 = get_closest_example(xvar_id_map, example, solutions[0].box())
        vfake = at.predict_single(example1)
        diff = min(example1 - example), max(example1 - example)
        print("diff", diff)

        print("predictions:", vreal, vfake, "(", solutions[0].output0, ")")
        print("num_steps:", opt.num_steps())
        self.assertTrue(abs(solutions[0].output0 - vfake) < 1e-5)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(example.reshape((28, 28)), cmap="binary")
        ax0.set_title(f"{vreal:.4f}")
        ax1.imshow(example1.reshape((28, 28)), cmap="binary")
        ax1.set_title(f"{vfake:.4f}")
        im = ax2.imshow((example1-example).reshape((28, 28)), cmap="binary")
        fig.colorbar(im, ax=ax2)
        plt.show()

    def test_one_out_of_k(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 2)
        t.split( t.left(t.root()), 1)
        t.split(t.right(t.root()), 1)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 3.0)
        t.set_leaf_value(t.right(t.right(t.root())), 4.0)
        t = at.add_tree();
        t.split(t.root(), 0)
        t.set_leaf_value( t.left(t.root()), 0.0)
        t.set_leaf_value(t.right(t.root()), 100.0)

        opt = Optimizer(minimize=at)
        opt.adjuster.add_one_out_of_k([0, 1, 2], True)

        opt.steps(50)
        print("\n".join([str(s.box()) for s in opt.solutions()]))
        print("num_box_checks", opt.num_box_checks())

        self.assertEqual(opt.num_solutions(), 3)

    def test_one_out_of_k2(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 2)
        t.set_leaf_value( t.left(t.root()), 0.0)
        t.set_leaf_value(t.right(t.root()), 100.0)
        t = at.add_tree();
        t.split(t.root(), 0)
        t.split( t.left(t.root()), 1)
        t.split(t.right(t.root()), 1)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 3.0)
        t.set_leaf_value(t.right(t.right(t.root())), 4.0)
        t = at.add_tree();
        t.split(t.root(), 3)
        t.set_leaf_value( t.left(t.root()), 0.0)
        t.set_leaf_value(t.right(t.root()), 100.0)
        t = at.add_tree();
        t.split(t.root(), 4)
        t.split( t.left(t.root()), 3)
        t.set_leaf_value(t.right(t.root()), 100.0)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)

        print(at)

        opt = Optimizer(minimize=at)
        opt.adjuster.add_one_out_of_k([0, 1, 2, 3, 4], True)

        opt.steps(50)
        print("\n".join([str(s.box()) for s in opt.solutions()]))

        self.assertEqual(opt.num_solutions(), 5)

    def test_less_than(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 0, 10.0) # X0 < 10.0
        t.split(t.left(t.root()), 1, 10.0) # X1 < 1.0
        t.split(t.right(t.root()), 1, 10.0) # X1 < 1.0
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 3.0)
        t.set_leaf_value(t.right(t.right(t.root())), 4.0)

        opt = Optimizer(minimize=at)
        opt.adjuster.add_less_than(0, 1, 0.0)

        opt.steps(50)
        sols = opt.solutions()
        print("\n".join([str((s.box(), s.output0)) for s in sols]))

        self.assertEqual(opt.num_solutions(), 3)
        self.assertEqual(sols[0].output0, 1.0)
        self.assertEqual(sols[1].output0, 2.0)
        self.assertEqual(sols[2].output0, 4.0) # skip 3

    def test_less_than2(self):
        at = AddTree()
        t = at.add_tree()
        t.split(t.root(), 0, 10.0)
        t.set_leaf_value(t.left( t.root()), 1.0) # X0 <  10.0
        t.set_leaf_value(t.right(t.root()), 2.0) # X0 >= 10.0
        t = at.add_tree()
        t.split(t.root(), 1, 10.0)
        t.set_leaf_value(t.left( t.root()), 10.0) # X1 <  10.0
        t.set_leaf_value(t.right(t.root()), 20.0) # X1 >= 10.0
        t = at.add_tree()
        t.split(t.root(), 1, 1.0)
        t.set_leaf_value(t.left( t.root()), 0.2) # X1 <  1.0
        t.set_leaf_value(t.right(t.root()), 0.4) # X1 >= 1.0

        opt = Optimizer(minimize=at)
        opt.adjuster.add_less_than(0, 1, 0.0)
        opt.steps(50)
        sols = opt.solutions()
        print("\n".join([str((s.box(), s.output0)) for s in sols]))

        self.assertEqual(opt.num_solutions(), 4)

        opt = Optimizer(minimize=at)
        opt.adjuster.add_less_than(1, 0, 0.0)
        opt.steps(50)
        sols = opt.solutions()
        print("\n".join([str((s.box(), s.output0)) for s in sols]))
        self.assertEqual(opt.num_solutions(), 5)



    def test_multithread(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis0-easy.json")
        opt = Optimizer(maximize=at)
        opt.steps(2)

        with opt.parallel(2) as paropt:
            for i in range(paropt.num_threads()):
                wopt = paropt.worker_opt(i)
                print(f"worker{i}: #candidates={wopt.num_candidate_cliques()}")

            paropt.steps_for(100)

        print("num_solutions", paropt.num_solutions())
        print("num_valid_solutions", paropt.num_new_valid_solutions())
        for i in range(paropt.num_threads()):
            wopt = paropt.worker_opt(i)
            print(f"worker{i}: #sol={wopt.num_solutions()}, #steps={wopt.num_steps}")

    def multithread2(self):
        #at = AddTree.read(f"tests/models/xgb-calhouse-hard.json")
        at = AddTree.read(f"tests/models/xgb-mnist-yis0-hard.json")

        # A*
        opt = Optimizer(maximize=at, use_dyn_prog_heuristic=False, max_memory=1024*1024*1024*1)
        print("num_vertices", opt.g1.num_vertices())
        start = timeit.default_timer()
        opt.steps(10)
        #while timeit.default_timer() - start < 20 and opt.opt.num_solutions() == 0:
        #    opt.steps(1000)
        #    print("opt", opt.current_bounds(), opt.opt.num_steps, opt.opt.num_box_checks)

        print(opt.opt.num_solutions())
        print(timeit.default_timer() - start, "sec")

        opt_a = Optimizer(maximize=at, use_dyn_prog_heuristic=False, max_memory=1024*1024*1024*1)
        opt_a.steps(10)
        steps_for_dur = 50
        try:
            start = timeit.default_timer()
            paropt_a = opt_a.parallel(4)
            while timeit.default_timer() - start < 60 and paropt_a.num_solutions() == 0:
                paropt_a.steps_for(steps_for_dur)
                steps_for_dur = min(1000, int(steps_for_dur * 1.5))
                for i in range(paropt_a.num_threads()):
                    wopt = paropt_a.worker_opt(i)
                    print(f"worker{i}", wopt.current_bounds(), wopt.num_steps, wopt.num_box_checks)
        finally:
            paropt_a.join_all()

        print(timeit.default_timer() - start, "sec", paropt_a.num_solutions())

        # ARA*
        opt_ara = Optimizer(maximize=at, ara_eps=0.5, max_memory=1024*1024*1024*1)
        opt_ara.steps(10)
        print("opt_ara", opt_ara.opt.num_candidate_cliques(), opt_ara.num_solutions(), opt_ara.opt.get_eps())

        steps_for_dur = 10
        new_eps = opt_ara.get_eps()
        try:
            start = timeit.default_timer()
            paropt_ara = opt_ara.parallel(4)
            while timeit.default_timer() - start < 60 \
                    and not (paropt_ara.get_eps() == 1.0 and paropt_ara.num_new_valid_solutions() > 0) \
                    and paropt_ara.num_candidate_cliques() > 0:
                if paropt_ara.num_new_valid_solutions() > 0:
                    new_eps = new_eps + (1.0 - new_eps) / 10 if new_eps < 0.99 else 1.0
                    print(f"EPS: {paropt_ara.get_eps()} -> {new_eps}")
                    paropt_ara.set_eps(new_eps)
                    steps_for_dur = 10

                paropt_ara.steps_for(steps_for_dur)
                steps_for_dur  = min(1000, int(steps_for_dur * 1.5))

                #for i in range(paropt_ara.num_threads()):
                #    wopt = paropt_ara.worker_opt(i)
                #    num_cands = wopt.num_candidate_cliques()
                #    num_sols = wopt.num_solutions()
                #    eps = wopt.get_eps()
                #    print(f"worker{i}: #candidates={num_cands}, #sol={num_sols}, eps={eps}")
        finally:
            paropt_ara.join_all()

        print(timeit.default_timer() - start, "sec")

        fig, ax = plt.subplots()

        best = -1000
        if paropt_a.num_solutions() > 0:
            best = max([s.output1 for s in paropt_a.solutions()])
            print("solution:", best)
        print(max([x[1] for x in paropt_a.bounds]))
        aresult = list(filter(lambda x: x[1][1] > best, zip(paropt_a.times, paropt_a.bounds)))
        t0 = [x[0] for x in aresult]
        s0 = [x[1][1] for x in aresult]
        l0, = ax.plot(t0, s0, label="a* upper")
        if paropt_a.num_solutions() > 0:
            ax.axhline(best, linestyle="--", color=l0.get_color(), label="a* best")
        sols = filter_solutions(paropt_ara)
        print("out", list(map(lambda s: round(s.output1, 2), sols)))
        print("eps", list(map(lambda s: round(s.eps, 2), sols)))
        print("up ", list(map(lambda s: round(s.output1 / s.eps, 2), sols)))
        print("valid", list(map(lambda s: s.is_valid, sols)))
        t1 = [s.time for s in sols]
        l1, = ax.plot(t1, [s.output1 for s in sols], '.', label="ara* lower")
        l2, = ax.plot(t1, [s.output1 / s.eps for s in sols], label="ara* upper")
        ax.plot([s.time for s in paropt_ara.solutions() if s.is_valid], [s.output1 for s in paropt_ara.solutions() if s.is_valid], 'v', markersize=1.5, label="ara* valid")
        ax.plot([s.time for s in paropt_ara.solutions() if not s.is_valid], [s.output1 for s in paropt_ara.solutions() if not s.is_valid], 'x', markersize=5, label="ara* invalid")
        ax.axhline(max(map(lambda s: s.output1, sols)), linestyle="--", color=l1.get_color(), label="ara* best")
        araresult = list(zip(paropt_ara.times, paropt_ara.bounds))
        t1 = [x[0] for x in araresult]
        s1 = [x[1][1] for x in araresult]
        ax.plot(t1, s1, ':', label="ara* bounds")
        ax.legend()
        plt.show()

    def merge_basic_bound(self):
        at = AddTree.read(f"tests/models/xgb-calhouse-easy.json")
        opt0 = Optimizer(minimize=at)
        opt1 = Optimizer(maximize=at)

        opt0.merge(3, reset_optimizer=False)
        opt1.merge(3, reset_optimizer=False)

        print(opt0.g0.num_independent_sets())
        print(opt1.g1.num_independent_sets())

        print(opt0.current_basic_bounds())
        print(opt1.current_basic_bounds())

    def test_add_negated(self):
        dummy = AddTree()
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 8.0)

        finfo = FeatInfo(dummy, at, set(), False);
        g0 = KPartiteGraph(at, finfo, 0)
        g1 = KPartiteGraph(at, finfo, 0)
        g0.add_with_negated_leaf_values(g1)

        g0.merge(2)
        self.assertEqual((0.0, 0.0), g0.basic_bound())

    #def test_simplify(self):
    #    at = AddTree()
    #    t = at.add_tree();
    #    t.split(t.root(), 0, 2)
    #    t.split( t.left(t.root()), 0, 1)
    #    t.split(t.right(t.root()), 0, 3)
    #    t.set_leaf_value( t.left( t.left(t.root())), 8.0)
    #    t.set_leaf_value(t.right( t.left(t.root())), 4.0)
    #    t.set_leaf_value( t.left(t.right(t.root())), 2.0)
    #    t.set_leaf_value(t.right(t.right(t.root())), 1.0)

    #    opt = Optimizer(maximize=at, simplify=(2.0, 1, False)) # underestimate
    #    self.assertEqual(opt.num_vertices(1), 3)

    #def test_simplify2(self):
    #    at = AddTree()
    #    t = at.add_tree();
    #    t.split(t.root(), 0, 5)
    #    t.split(t.right(t.root()), 1, 5)
    #    t.split(t.left(t.right(t.root())), 2, 5)
    #    t.set_leaf_value(t.left(t.root()), 1)
    #    t.set_leaf_value( t.left(t.left(t.right(t.root()))), 2)
    #    t.set_leaf_value(t.right(t.left(t.right(t.root()))), 3)
    #    t.set_leaf_value(t.right(t.right(t.root())), 4)

    #    #print(at)

    #    opt = Optimizer(minimize=at, simplify=(2.0, 0, True)) # overestimate
    #    self.assertEqual(opt.num_vertices(0), 2)
    #    self.assertFalse(opt.step(10))
    #    self.assertEqual(opt.solutions()[0][0], 1)
    #    self.assertEqual(opt.solutions()[1][0], 4)

    #def test_simplify3(self):
    #    with open("tests/models/xgb-img-very-easy-values.json") as f:
    #        ys = json.load(f)
    #    imghat = np.array(ys).reshape((100, 100))
    #    at = AddTree.read(f"tests/models/xgb-img-very-easy.json")
    #    print(at)

    #    opt0 = Optimizer(minimize=at)
    #    opt1 = Optimizer(minimize=at, simplify=(50.0, 0, True)) # overestimate

    #    opt0.step(100)
    #    opt1.step(100)

    #    print(opt0.solutions()[0])
    #    print(opt1.solutions()[0])

    #    self.assertTrue(opt1.solutions()[0][0] - opt0.solutions()[0][0] < 50.0)

    #    #plot_img_solutions(imghat, opt0.solutions()[0:1])
    #    #plot_img_solutions(imghat, opt1.solutions()[0:1])

    #def test_simplify4(self):
    #    with open("tests/models/xgb-img-easy-values.json") as f:
    #        ys = json.load(f)
    #    imghat = np.array(ys).reshape((100, 100))
    #    at = AddTree.read(f"tests/models/xgb-img-easy.json")

    #    def numvert(opt0, opt1):
    #        v0 = opt0.num_vertices(0)
    #        v1 = opt1.num_vertices(0)
    #        return v0, v1, (v0-v1)/v0

    #    opt0 = Optimizer(minimize=at)
    #    opt1 = Optimizer(minimize=at, simplify=(20.0, 0, True)) # overestimate

    #    print("num_vertices", numvert(opt0, opt1))
    #    opt0.prune([70, 50], 20.0)
    #    opt1.prune([70, 50], 20.0)
    #    print("num_vertices", numvert(opt0, opt1))

    #    #opt0.set_ara_eps(0.5, 0.1)
    #    while opt0.num_solutions() == 0:
    #        opt0.step(1)
    #    #opt1.set_ara_eps(0.5, 0.1)
    #    while opt1.num_solutions() == 0:
    #        opt1.step(1)

    #    print(opt0.solutions())
    #    print(opt1.solutions())

    #    print(opt0.nsteps())
    #    print(opt1.nsteps())

    #    plot_img_solutions(imghat, opt0.solutions()[0:1])
    #    plot_img_solutions(imghat, opt1.solutions()[0:1])


if __name__ == "__main__":
    #z3.set_pp_option("rational_to_decimal", True)
    #z3.set_pp_option("precision", 3)
    #z3.set_pp_option("max_width", 130)

    unittest.main()
