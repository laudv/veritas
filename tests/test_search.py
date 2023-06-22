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
        at = AddTree(1)
        t = at.add_tree();
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

    def test_img1(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy-new.json"))
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))
        config = Config(HeuristicType.MAX_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

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

    def test_img1_min(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy-new.json"))
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))
        config = Config(HeuristicType.MIN_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

        done = StopReason.NONE
        while done == StopReason.NONE:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)
        self.assertEqual(search.num_solutions(), 32);

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output <= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        self.assertAlmostEqual(solutions[0].output, m, 4)
        self.assertAlmostEqual(solutions[-1].output, M, 4)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def test_img2(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))
        at.set_base_score(0, at.get_base_score(0) - np.median(y))
        #at = at.prune([Domain(0, 30), Domain(0, 30)])
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        m, M = min(yhat), max(yhat)
        #img = np.array(ys).reshape((100, 100))

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
        self.assertEqual(search.num_solutions(), len(outputs_expected));

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertTrue(all(x.output >= y.output for x, y in zip(solutions[:-1], solutions[1:]))) #sorted?
        for s, x in zip(solutions, outputs_expected):
            self.assertLess(abs(s.output-x)/x, 1e-4)

        plot_img_solutions(imghat, solutions[:3])
        plot_img_solutions(imghat, solutions[-3:])

    def _do_img_multiclass(self, at, heuristic, class_opt):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in np.linspace(0, 100, 251)[:-1]
                             for y in np.linspace(0, 100, 251)[:-1]])
        y = np.array([img[int(x), int(y)] for x, y in X])
        yc = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
        X = X.astype(np.float32)

        for cls in range(4):
            at.swap_class(cls);
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
            solution_outputs = np.unique([sol.output for sol in solutions])

            covered = np.zeros((1000, 1000))

            for i, sol in enumerate(solutions):
                box = sol.box()
                box[0] = box.get(0, Interval(0, 100)).intersect(Interval(0, 100))
                box[1] = box.get(1, Interval(0, 100)).intersect(Interval(0, 100))
                covered[int(box[0].lo*10):int(box[0].hi*10),
                        int(box[1].lo*10):int(box[1].hi*10)] += 1
                ex = get_closest_example(box, np.zeros(2))
                pred = at.eval(ex)[0]
                expected = pred[0] - class_opt(pred[1:])
                self.assertAlmostEqual(sol.output, expected)

            self.assertTrue(np.all(covered == 1.0))
            self.assertTrue(np.max(np.abs(solution_outputs - output_exp_sorted)) < 1e-10)


            #plot_img_solutions(yc.reshape((100, 100)), solutions[:3])
            #plot_img_solutions(yhat[:, 0].reshape((100, 100)), solutions[:3])
            #plot_img_solutions(output_exp.reshape((250, 250)), solutions[:10])
            #at.swap_class(cls); # back to normal

    def test_img_multiclass(self):
        print("XGB MAX_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MAX_MAX_OUTPUT_DIFF, np.max)

        print("RF MAX_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MAX_MAX_OUTPUT_DIFF, np.max)

        print("XGB MAX_MIN")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MAX_MIN_OUTPUT_DIFF, np.min)

        print("RF MAX_MIN")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MAX_MIN_OUTPUT_DIFF, np.min)

        print("XGB MIN_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MIN_MAX_OUTPUT_DIFF, np.max)

        print("RF MIN_MAX")
        at = AddTree.read(os.path.join(BPATH, "models/rf-img-multiclass.json"))
        self._do_img_multiclass(at, HeuristicType.MULTI_MIN_MAX_OUTPUT_DIFF, np.max)

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
        print("evaluate", ypred)
        rob = VeritasRobustnessSearch(None, at, example, start_delta=15)
        rob.search()

        ypred = at.eval(rob.generated_examples)
        self.assertTrue(ypred[-1] >= 0.0)

        try:
            kan = KantchelianAttack(at, True, example)
            kan.optimize()
            print(kan.solution())
        except Exception as e:
            print("Gurobi error!", e)

   # def test_img6(self):
   #     img = np.load(os.path.join(BPATH, "data/img.npy"))
   #     X = np.array([[x, y] for x in range(100) for y in range(100)])
   #     y = np.array([img[x, y] for x, y in X])
   #     ymed = np.median(y)
   #     X = X.astype(np.float32)

   #     at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))
   #     at.base_score -= ymed
   #     yhat = at.eval(X)
   #     imghat = np.array(yhat).reshape((100, 100))


   #     example = [70, 50]
   #     ypred = at.eval(example)[0]
   #     self.assertTrue(ypred < 0.0)
   #     print("evaluate", ypred)
   #     search = Search.min_dist_to_example(at, example, 0.0)
   #     print(search.step_for(1.0, 100))

   #     solutions = [search.get_solution(i) for i in range(search.num_solutions())]
#  #      for i, sol in enumerate(solutions):
#  #          print(at.eval([d.lo for d in sol.box().values()])[0],
#  #                  sol.output,
#  #                  search.get_solstate_field(i, "g"),
#  #                  search.get_solstate_field(i, "dist"),
#  #                  sol.box(), example, ypred, at.eval(get_closest_example(sol.box(), example)))
#
   #     print(search.current_bounds())

   #     #plot_img_solutions(imghat, solutions)

   #     #ypred = at.eval(rob.generated_examples)
   #     #self.assertTrue(ypred[-1] >= 0.0)

   #     #kan = KantchelianAttack(at, True, example)
   #     #kan.optimize()
   #     #print(kan.solution())

if __name__ == "__main__":
    unittest.main()
