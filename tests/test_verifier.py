import os
import math
import unittest
import numpy as np
import z3

from veritas import Interval, FloatT, AddTreeType, AddTree, Config, StopReason, \
        HeuristicType
from veritas.smt import Verifier#, not_in_domain_constraint, in_domain_constraint
from veritas.z3backend import Z3Backend as Backend
from veritas import KantchelianOutputOpt
from veritas import SMTRobustnessSearch, VeritasRobustnessSearch
from veritas import MilpRobustnessSearch

### NOTE: some tests were taken from treeck, and adapted to discard useless stuff (DomTree, etc.)
BPATH = os.path.dirname(__file__)

class TestVerifier(unittest.TestCase):

    def myAssertAlmostEqual(self, a, b, eps=1e-6):
        self.assertTrue(type(a) == type(b))
        if isinstance(a, list) or isinstance(a, tuple) or isinstance(a, np.ndarray):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.myAssertAlmostEqual(x, y, eps=eps)
        elif isinstance(a, float) or isinstance(a, FloatT) or isinstance(a, np.float64):
            self.assertAlmostEqual(a, b, delta=eps)
        else:
            self.assertEqual(a, b)


    def test_single_tree(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 0, 2)
        t.split(t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value(t.left(t.left(t.root())), 0.1)
        t.set_leaf_value(t.right(t.left(t.root())), 0.2)
        t.set_leaf_value(t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        v = Verifier(at, {}, Backend())
        #print(v._xvars, v._wvars, v._fvar)

        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.SAT)

        v.add_tree(0)
        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.41), Verifier.Result.UNSAT)

        prune_box = {0: Interval(1, 3)}
        v = Verifier(at, prune_box, Backend())
        v.add_all_trees()
        v.add_constraint(v.xvar(0) < 2.0)

        ### value for class 0 manually read here!
        check = v.check(v.fvar() != t.get_leaf_value(t.right(t.left(t.root())), 0)) 
        self.assertEqual(check, Verifier.Result.UNSAT)

        prune_box = {0: Interval(1, 3)}
        v = Verifier(at, prune_box, Backend())
        v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() > 0.41), Verifier.Result.UNSAT)
   

    def test_two_trees(self):
        at = AddTree(1, AddTreeType.REGR)
        t = at.add_tree()
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)
        t = at.add_tree()
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 1, 1)
        t.split(t.right(t.root()), 1, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), -0.3)
        t.set_leaf_value(t.right(t.right(t.root())), -0.4)

        v = Verifier(at, {}, Backend())
        v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.11), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > -0.09), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() > 0.41), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        v = Verifier(at, {}, Backend()); v.add_all_trees()
        v.add_constraint(v.xvar(0) < 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        v = Verifier(at, {}, Backend()); v.add_all_trees()
        v.add_constraint(v.xvar(0) >= 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.UNSAT)

        prune_box = {0: Interval(-math.inf, 2.0)}
        v = Verifier(at, prune_box, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        prune_box = {0: Interval(2.0, math.inf)}
        v = Verifier(at, prune_box, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.UNSAT)
        v.add_constraint(v.xvar(1) < 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() < 0.01), Verifier.Result.SAT)

        ### WIP 
        model = v.model()
        self.myAssertAlmostEqual(model["ws"], [0.3, -0.3])
        self.assertEqual(v.model_family(model), {0: Interval(2, 3),
                                                 1: Interval(-math.inf, 3)})
        #v.add_constraint(not_in_domain_constraint(v, v.model_family(model), 0))
        #self.assertEqual(v.check(v.fvar() < 0.01), Verifier.Result.UNSAT)


    def test_img(self):
        # new version of image loading from veritas tests
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))

        X = np.array([[x, y] for x in range(100) for y in range(100)])
        #y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))
        m, M = np.min(yhat), np.max(yhat)
        #print(m, M)
        # m<0: the ensemble can predict values <0

        # actual test
        #print("< 0")
        #dt = DomTree(at, {})
        #l0 = dt.get_leaf(dt.tree().root())
        #v = Verifier(l0, Backend()); v.add_all_trees()
        v = Verifier(at, {}, Backend())
        v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < 0.0), Verifier.Result.SAT)
        model = v.model()
        self.assertLess(model["f"], 0.0)
        self.assertGreaterEqual(model["f"], m)

        # this is from an old test - cannot work unless we enumerate all possible outputs
        #print("< m, > M")
        #self.assertEqual(v.check((v.fvar() < m)), Verifier.Result.UNSAT)
        #self.assertEqual(v.check((v.fvar() < m) | (v.fvar() > M)), Verifier.Result.UNSAT)


    ### NOTE: deleted a lot of tests on images from "treeck" - have a look if they're useful


    def test_img_max_output(self):
        # test where we compare result with SMT, Veritas, KantchelianOutputOpt

        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-very-easy-new.json"))
        yhat = at.eval(X).ravel()
        #imghat = np.array(yhat).reshape((100, 100))
        m, M = min(yhat), max(yhat) # min and max possible outputs

        # get all possible solutions (sorted from max to min output)
        config = Config(HeuristicType.MAX_OUTPUT)
        config.stop_when_optimal = False
        search = config.get_search(at)

        done = StopReason.NONE
        while done == StopReason.NONE:
            done = search.steps(100)
        self.assertTrue(done == StopReason.NO_MORE_OPEN)
        self.assertEqual(search.num_solutions(), 32)

        solutions = [search.get_solution(i) for i in range(search.num_solutions())]
        self.assertAlmostEqual(solutions[0].output, M, 4)
        self.assertAlmostEqual(solutions[-1].output, m, 4)
        #print([x.output for x in solutions])

        # 1. test SMT returns same result as Veritas MAX_OUTPUT 
        # set SMT constraint as: output > output of second-best solution
        # ---> SMT must return optimal solution
        second_best_output = solutions[1].output
        v = Verifier(at, {}, Backend())
        v.add_all_trees()
        self.assertEqual(v.check(v.fvar() > second_best_output), Verifier.Result.SAT)
        model = v.model()
        self.assertAlmostEqual(model["f"], solutions[0].output)

        # TODO (lorenzo): fix this
        # 2. test KantchelianOutputOpt returns same result as Veritas MAX_OUTPUT 
        #milp = KantchelianOutputOpt(at, example)
        #milp.optimize()
        #milp_output, milp_intervals = milp.solution()
        #self.assertAlmostEqual(milp_output, solutions[0].output)
        #self.assertAlmostEqual(milp_output, model["f"])

        # 3. check that intervals from SMT and Veritas are the same
        # note: KantchelianOutputOpt returns stricter milp_intervals
        self.assertEqual(solutions[0].box(), v.model_family(model))


    def test_img_robustness_search(self):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(np.float32)
        ymed = np.median(y)

        at = AddTree.read(os.path.join(BPATH, "models/xgb-img-easy-new.json"))
        #print("base score: ", at.get_base_score(0))
        at.set_base_score(0, at.get_base_score(0) - ymed)
        #print("base score: ", at.get_base_score(0))
        yhat = at.eval(X)
        imghat = np.array(yhat).reshape((100, 100))

        example = [70, 50]
        ypred = at.eval(example)[0]
        self.assertTrue(ypred < 0.0)
        #print("evaluate", ypred)

        print()
        print("*** Veritas Robustness Search ***")
        rob = VeritasRobustnessSearch(example, 15, None, at)
        rob.search()

        ypred = at.eval(rob.generated_examples)
        self.assertTrue(ypred[-1] >= 0.0)

        try:
            print()
            print("*** MILP Robustness Search ***")

            kan = MilpRobustnessSearch(example, 15, None, at)
            kan.search()

            ypred = at.eval(kan.generated_examples)
            self.assertTrue(ypred[-1] >= 0.0)

            # result should be the same as Veritas
            self.myAssertAlmostEqual(rob.generated_examples[-1],
                                        kan.generated_examples[-1])
        except Exception as e:
            print("Gurobi error!", e)

        try:
            print()
            print("*** SMT Robustness Search ***")

            smt = SMTRobustnessSearch(example, 15, None, at)
            smt.search()

            # TODO (lorenzo): fix this
            # note: result not the exact same with SMT, but still SAT
            #ypred = at.eval(smt.generated_examples[-1])
            #self.assertTrue(ypred[-1] >= 0.0)

        except Exception as e:
            print("z3 error!", e)

        # TODO (lorenzo): fix this
        # in this easy case, output of solutions is exactly the same
        #self.myAssertAlmostEqual(at.predict(rob.generated_examples[-1]),
        #                            at.predict(kan.generated_examples[-1]))
        #self.myAssertAlmostEqual(at.predict(rob.generated_examples[-1]),
        #                            at.predict(smt.generated_examples[-1]))

  
if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
