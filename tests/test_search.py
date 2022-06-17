import unittest, sys, os
import numpy as np

try:
    MATPLOTLIB=True
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ModuleNotFoundError as e:
    MATPLOTLIB=False

from veritas import *

BPATH = os.path.dirname(__file__)

def plot_img_solutions(imghat, solutions):
    if not MATPLOTLIB:
        return
    fig, ax = plt.subplots()
    im = ax.imshow(imghat)
    fig.colorbar(im, ax=ax)
    i, j, c = 1, 0, "r"
    for s in solutions:
        box = s.box()
        x0, y0 = max(0.0, box[i].lo), max(0.0, box[j].lo)
        x1, y1 = min(100.0, box[i].hi), min(100.0, box[j].hi)
        w, h = x1-x0, y1-y0
        #print((x0, y0), (x1, y1), w, h, s.output)
        rect = patches.Rectangle((x0-0.5,y0-0.5),w,h,lw=1,ec=c,fc='none')
        ax.add_patch(rect)

    plt.show()

class TestSearch(unittest.TestCase):
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
        print(at[0])

        search = Search.max_output(at)
        search.debug = True

        done = False
        while not done:
            done = search.steps(100)

        print("num_rej", search.num_rejected_solutions)

        print(done, search.num_solutions(), search.get_solution(0))
        self.assertTrue(done)
        self.assertEqual(search.num_solutions(), 4)
        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertEqual(solutions[0].output, 8.0)
        self.assertEqual(solutions[1].output, 4.0)
        self.assertEqual(solutions[2].output, 2.0)
        self.assertEqual(solutions[3].output, 1.0)
        self.assertEqual(solutions[0].box()[0], Domain.from_lo(3))
        self.assertEqual(solutions[1].box()[0], Domain.exclusive(2, 3))
        self.assertEqual(solutions[2].box()[0], Domain.exclusive(1, 2))
        self.assertEqual(solutions[3].box()[0], Domain.from_hi_exclusive(1))

    def test_img1(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy.json"))
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))

        search = Search.max_output(at)
        search.stop_when_optimal = False
        done = StopReason.NONE
        while done == StopReason.NONE:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)
        self.assertEqual(search.num_solutions(), 32);

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        self.assertAlmostEqual(solutions[0].output, M, 4)
        self.assertAlmostEqual(solutions[-1].output, m, 4)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def test_img2(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
        at.base_score -= np.median(y)
        #at = at.prune([Domain(0, 30), Domain(0, 30)])
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))

        search = Search.max_output(at)
        search.stop_when_optimal = False
        done = StopReason.NONE
        search.prune([Domain(0, 30), Domain(0, 30)])
        while done != StopReason.NO_MORE_OPEN:
            done = search.steps(100)
            #print("done?", done)
            #print(search.current_bounds(), search.is_optimal())
            #print(search.snapshots[-1].avg_focal_size)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)

        outputs_expected = sorted(np.unique(imghat[0:30, 0:30]), reverse=True)
        self.assertEqual(search.num_solutions(), len(outputs_expected));

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        for s, x in zip(solutions, outputs_expected):
            self.assertLess(abs(s.output-x)/x, 1e-4)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

        #for i, sol in enumerate(solutions):
        #    print(at.eval([d.lo for d in sol.box().values()])[0],
        #            sol.output,
        #            search.get_solstate_field(i, "g"),
        #            search.get_solstate_field(i, "h"))

    #def test_img4(self):
    #    img = np.load(os.path.join(BPATH, "data/img.npy"))
    #    X = np.array([[x, y] for x in range(100) for y in range(100)])
    #    y = np.array([img[x, y] for x, y in X])
    #    ymed = np.median(y)
    #    X = X.astype(np.float32)

    #    at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
    #    at.base_score -= ymed
    #    yhat = at.eval(X)
    #    imghat = np.array(yhat).reshape((100, 100))


    #    example = [70, 50]
    #    print("evaluate", at.eval(example)[0])
    #    search = GraphRobustnessSearch(at, example, 15)
    #    search.stop_when_num_solutions_equals = 1
    #    done = search.step_for(1000, 100)
    #    done = search.steps(100)
    #    done = search.steps(100)
    #    done = search.steps(100)
    #    done = search.steps(100)

    #    print("done?", done, "num_sol?", search.num_solutions(), search.num_steps())
    #    plot_img_solutions(imghat, [search.get_solution(i) for i in range(search.num_solutions())])


    def test_img5(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        ymed = np.median(y)
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
        at.base_score -= ymed
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))


        example = [70, 50]
        ypred = at.eval(example)[0]
        self.assertTrue(ypred < 0.0)
        print("evaluate", ypred)
        rob = VeritasRobustnessSearch(None, at, example, start_delta=15)
        rob.search()

        ypred = at.eval(rob.generated_examples)
        self.assertTrue(ypred[-1] >= 0.0)

        try:
            kan = KantchelianAttack(at, True, example)
            kan.optimize()
            print(kan.solution())
        except:
            print("Gurobi error!")

    def test_img6(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        ymed = np.median(y)
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
        at.base_score -= ymed
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))


        example = [70, 50]
        ypred = at.eval(example)[0]
        self.assertTrue(ypred < 0.0)
        print("evaluate", ypred)
        search = Search.min_dist_to_example(at, example, 0.0)
        print(search.step_for(1.0, 100))

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
#        for i, sol in enumerate(solutions):
#            print(at.eval([d.lo for d in sol.box().values()])[0],
#                    sol.output,
#                    search.get_solstate_field(i, "g"),
#                    search.get_solstate_field(i, "dist"),
#                    sol.box(), example, ypred, at.eval(get_closest_example(sol.box(), example)))
#
        print(search.current_bounds())

        #plot_img_solutions(imghat, solutions)

        #ypred = at.eval(rob.generated_examples)
        #self.assertTrue(ypred[-1] >= 0.0)

        #kan = KantchelianAttack(at, True, example)
        #kan.optimize()
        #print(kan.solution())

if __name__ == "__main__":
    unittest.main()
