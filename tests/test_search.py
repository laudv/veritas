import unittest
import os
import imageio.v3 as imageio
import numpy as np

try:
    MATPLOTLIB=True
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ModuleNotFoundError:
    MATPLOTLIB=False

import veritas
from veritas import AddTree, AddTreeType, HeuristicType, Config, \
        Interval, StopReason, VeritasRobustnessSearch

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

def get_img_data():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    yr = np.array([img[x, y] for x, y in X]).astype(float)
    y2 = yr > np.median(yr) # binary clf
    y4 = np.digitize(yr, np.quantile(yr, [0.25, 0.5, 0.75])) # multiclass
    X = X.astype(float)

    return img, X, yr, y2, y4

class TestSearch(unittest.TestCase):
    def test_single_tree(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0, 1.0)
        t.set_leaf_value(t.right( t.left(t.root())), 0, 2.0)
        t.set_leaf_value( t.left(t.right(t.root())), 0, 4.0)
        t.set_leaf_value(t.right(t.right(t.root())), 0, 8.0)
        print(at[0])

        config = Config(HeuristicType.MAX_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

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
        self.assertEqual(solutions[0].box()[0], Interval.from_lo(3))
        self.assertEqual(solutions[1].box()[0], Interval(2, 3))
        self.assertEqual(solutions[2].box()[0], Interval(1, 2))
        self.assertEqual(solutions[3].box()[0], Interval.from_hi(1))

    def test_img1_max(self):
        img, X, y, _, _ = get_img_data()

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy-new.json"))
        yhat = at.eval(X).ravel()

        m, M = min(yhat), max(yhat)
        config = Config(HeuristicType.MAX_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

        done = StopReason.NONE
        while done == StopReason.NONE:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)
        self.assertEqual(search.num_solutions(), 32)

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y
                            in zip(solutions[:-1], solutions[1:]))) #sorted?
        print(solutions[0].output, M)
        self.assertAlmostEqual(solutions[0].output, M, 4)
        self.assertAlmostEqual(solutions[-1].output, m, 4)

        #imghat = np.array(yhat).reshape((100, 100))
        #plot_img_solutions(imghat, solutions[:3])
        #plot_img_solutions(imghat, solutions[-3:])

    def test_img1_min(self):
        img, X, y, _, _ = get_img_data()

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy-new.json"))
        yhat = at.eval(X).ravel()

        m, M = min(yhat), max(yhat)
        config = Config(HeuristicType.MIN_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

        done = StopReason.NONE
        while done == StopReason.NONE:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)
        self.assertEqual(search.num_solutions(), 32)

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output <= y.output for x, y
                            in zip(solutions[:-1], solutions[1:]))) #sorted?
        self.assertAlmostEqual(solutions[0].output, m, 4)
        self.assertAlmostEqual(solutions[-1].output, M, 4)

        #imghat = np.array(yhat).reshape((100, 100))
        #plot_img_solutions(imghat, solutions[:3])
        #plot_img_solutions(imghat, solutions[-3:])

    def test_img2(self):
        img, X, y, _, _ = get_img_data()

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))
        #at = at.prune([Domain(0, 30), Domain(0, 30)])
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        config = Config(HeuristicType.MAX_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at, [Interval(0, 30), Interval(0, 30)])

        done = StopReason.NONE
        while done != StopReason.NO_MORE_OPEN:
            done = search.steps(100)
            #print("done?", done)
            #print(search.current_bounds(), search.is_optimal())
            #print(search.snapshots[-1].avg_focal_size)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)

        outputs_expected = sorted(np.unique(imghat[0:30, 0:30]), reverse=True)
        self.assertEqual(search.num_solutions(), len(outputs_expected))

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y
                            in zip(solutions[:-1], solutions[1:]))) #sorted?
        for s, x in zip(solutions, outputs_expected):
            self.assertLess(abs(s.output-x)/x, 1e-4)

        #plot_img_solutions(imghat, solutions[:3])
        #plot_img_solutions(imghat, solutions[-3:])

    def _do_img_multiclass(self, at, heuristic, output_opt, class_opt):
        img, X, y, yc, _ = get_img_data()

        class_opt = np.max if class_opt=="max" else np.min

        for cls in range(4):
            at.swap_class(cls)
            yhat = at.eval(X)

            config = Config(heuristic)
            config.stop_when_optimal = False
            #config.ignore_state_when_worse_than = 0.0
            search = config.get_search(at)#, [Interval(10, 30), Interval(10, 30)])

            done = StopReason.NONE
            while done != StopReason.NO_MORE_OPEN:
                done = search.steps(100)
            self.assertTrue(done == StopReason.NO_MORE_OPEN)

            print(f"class={cls}, {done}", end="")
            print(f", time {search.time_since_start():.3f}", end="")
            print(f", #ignored {search.stats.num_states_ignored}", end="")
            print(f", #steps {search.stats.num_steps}", end="")
            print(f", #sols {search.num_solutions()}")

            solutions = [search.get_solution(i) for i in range(search.num_solutions())]
            output_exp = yhat[:, 0] - class_opt(yhat[:, 1:], axis=1)
            output_exp_sorted = np.unique(output_exp)
            if output_opt == "max":
                output_exp_sorted = output_exp_sorted[::-1]
            k = 0 # index into output_exp_sorted

            covered = np.zeros((1000, 1000), dtype=int)

            for i, sol in enumerate(solutions):
                box = sol.box()
                box[0] = box.get(0, Interval(0, 100)).intersect(Interval(0, 100))
                box[1] = box.get(1, Interval(0, 100)).intersect(Interval(0, 100))
                covered[int(box[0].lo*10):int(box[0].hi*10),
                        int(box[1].lo*10):int(box[1].hi*10)] += 1
                ex = veritas.get_closest_example(box, np.zeros(2), 1e-5)
                pred = at.eval(ex)[0]
                expected = pred[0] - class_opt(pred[1:])
                self.assertAlmostEqual(sol.output, expected)

                # robust to small numeric differences
                if output_opt == "max":
                    while sol.output < output_exp_sorted[k] - 1e-9:
                        k += 1
                else:
                    while sol.output > output_exp_sorted[k] + 1e-9:
                        k += 1

                if not box[0].contains(round(box[0].lo)) \
                        or not box[1].contains(round(box[1].lo)):
                    #print("skip", box[0], box[1], "-- not an integer solution")
                    continue

                #print(i, k, sol.output, output_exp_sorted[k],
                #      box[0], box[0].contains(round(box[0].lo)),
                #      box[1], box[1].contains(round(box[1].lo)))
                self.assertAlmostEqual(sol.output, output_exp_sorted[k])

            # must all be equal to 1
            # BAD: equal to 0 -> we skipped a part of the solution space
            # BAD: > 1        -> there were overlapping solutions
            self.assertTrue(np.all(covered == 1))

            # if not equal, then we did not enumerate all solutions
            self.assertEqual(k + 1, len(output_exp_sorted))
            self.assertAlmostEqual(solutions[0].output, output_exp_sorted[0])

            at.swap_class(cls) # back to normal

    def test_img_multiclass(self):
        HType = HeuristicType

        print("XGB MAX_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MAX_MAX_OUTPUT_DIFF, "max", "max")

        print("XGB MULTIVALUE MAX_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass-multivalue.json"))
        self._do_img_multiclass(at, HType.MULTI_MAX_MAX_OUTPUT_DIFF, "max", "max")

        print("RF MAX_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MAX_MAX_OUTPUT_DIFF, "max", "max")

        print("XGB MAX_MIN")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MAX_MIN_OUTPUT_DIFF, "max", "min")

        print("RF MAX_MIN")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MAX_MIN_OUTPUT_DIFF, "max", "min")

        print("XGB MIN_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MIN_MAX_OUTPUT_DIFF, "min", "max")

        print("RF MIN_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HType.MULTI_MIN_MAX_OUTPUT_DIFF, "min", "max")

    def test_img_multiclass_invalidate(self):
        img, X, y, yc, _ = get_img_data()
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))

        config = Config(HeuristicType.MULTI_MAX_MAX_OUTPUT_DIFF)
        config.stop_when_optimal = False
        config.multi_ignore_state_when_class0_worse_than = 6.0
        search = config.get_search(at)

        done = StopReason.NONE
        while done != StopReason.NO_MORE_OPEN:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)

        print(f"class=0, {done}", end="")
        print(f", time {search.time_since_start():.3f}", end="")
        print(f", #ignored {search.stats.num_states_ignored}", end="")
        print(f", #update_score_fails {search.stats.num_update_scores_fails}", end="")
        print(f", #steps {search.stats.num_steps}", end="")
        print(f", #sols {search.num_solutions()}")

        for i in range(search.num_solutions()):
            sol = search.get_solution(i)
            ex = veritas.get_closest_example(sol.box(), np.zeros(2), 1e-5)
            pred = at.eval(ex)[0]
            expected = pred[0] - np.max(pred[1:])
            self.assertAlmostEqual(sol.output, expected)
            self.assertTrue(pred[0] >=
                            config.multi_ignore_state_when_class0_worse_than)

        # Same search, but output for class0 unconstrained
        # The number of solutions with prob(class0) > {value} must be equal
        config2 = Config(HeuristicType.MULTI_MAX_MAX_OUTPUT_DIFF)
        config2.stop_when_optimal = False
        search2 = config2.get_search(at)

        done = StopReason.NONE
        while done != StopReason.NO_MORE_OPEN:
            done = search2.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)

        count = 0
        for i in range(search2.num_solutions()):
            sol = search2.get_solution(i)
            ex = veritas.get_closest_example(sol.box(), np.zeros(2), 1e-5)
            pred = at.eval(ex)[0]
            expected = pred[0] - np.max(pred[1:])
            self.assertAlmostEqual(sol.output, expected)

            if pred[0] >= config.multi_ignore_state_when_class0_worse_than:
                count += 1

        self.assertEqual(count, search.num_solutions())





    def test_img5(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        ymed = np.median(y)
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))
        at.set_base_score(0, at.get_base_score(0) - ymed)
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        example = [70, 50]
        ypred = at.eval(example)[0]
        self.assertTrue(ypred < 0.0)
        start_delta = 15
        rob = VeritasRobustnessSearch(example, start_delta, None, at)
        rob.search()

        ypred = at.eval(rob.generated_examples)
        self.assertTrue(ypred[-1] >= 0.0)

        try:
            kan = veritas.KantchelianAttack(at, True, example)
            kan.optimize()
            print(kan.solution())
        except Exception as e:
            print("Gurobi error!", e)


if __name__ == "__main__":
    unittest.main()
