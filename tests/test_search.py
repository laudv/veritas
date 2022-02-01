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
        print((x0, y0), (x1, y1), w, h, s.output)
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

        search = GraphOutputSearch(at)

        done = False
        while not done:
            done = search.steps(100)

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

        search = GraphOutputSearch(at)
        done = False
        while not done:
            done = search.steps(100)
        self.assertTrue(done)
        self.assertEqual(search.num_solutions(), 32);

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        self.assertAlmostEqual(solutions[0].output, M, 5)
        self.assertAlmostEqual(solutions[-1].output, m, 5)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def test_img2(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
        at.base_score = 0.0
        #at = at.prune([Domain(0, 30), Domain(0, 30)])
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))

        search = GraphOutputSearch(at)
        search.prune([Domain(0, 30), Domain(0, 30)])
        done = False
        while not done:
            done = search.steps(100)

        self.assertTrue(done)
        outputs_expected = sorted(np.unique(imghat[0:30, 0:30]), reverse=True)
        self.assertEqual(search.num_solutions(), len(outputs_expected));

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        for s, x in zip(solutions, outputs_expected):
            self.assertLess((s.output-x)/x, 1e-5)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def test_img3(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy.json"))
        print(at.base_score)
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        at = at.neutralize_negative_leaf_values()
        print(at.base_score)
        yhat2 = at.eval(X)

        print("output difference", sum(abs(yhat-yhat2))/len(yhat))
        self.assertLess(sum(abs(yhat-yhat2))/len(yhat), 1e-5)

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))

        search = GraphOutputSearch(at)
        done = False
        while not done:
            done = search.steps(100)
        self.assertTrue(done)
        #self.assertEqual(search.num_solutions(), 32);

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        self.assertLess((solutions[0].output-M)/M, 1e-5)
        self.assertLess((solutions[-1].output-m)/m, 1e-5)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def test_img4(self):
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
        print("evaluate", at.eval(example))
        search = GraphRobustnessSearch(at, example, 15)
        search.stop_when_num_solutions_equals = 1
        done = search.step_for(1000, 100)
        done = search.steps(100)
        done = search.steps(100)
        done = search.steps(100)
        done = search.steps(100)

        print("done?", done, "num_sol?", search.num_solutions(), search.num_steps())
        plot_img_solutions(imghat, [search.get_solution(i) for i in range(search.num_solutions())])




        #at = at.neutralize_negative_leaf_values()
        #print(at.base_score)
        #yhat2 = at.eval(X)

        #print("output difference", sum(abs(yhat-yhat2))/len(yhat))
        #self.assertLess(sum(abs(yhat-yhat2))/len(yhat), 1e-5)

        #m, M = min(yhat), max(yhat)
        ##img = np.array(ys).reshape((100, 100))

        #search = GraphOutputSearch(at)
        #done = False
        #while not done:
        #    done = search.steps(100)
        #self.assertTrue(done)
        ##self.assertEqual(search.num_solutions(), 32);

        #solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        #self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        #self.assertLess((solutions[0].output-M)/M, 1e-5)
        #self.assertLess((solutions[-1].output-m)/m, 1e-5)

        #plot_img_solutions(imghat, solutions[:3])
        #plot_img_solutions(imghat, solutions[-3:])

if __name__ == "__main__":
    unittest.main()
