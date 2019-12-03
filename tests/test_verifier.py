import matplotlib.pyplot as plt
import unittest, json
import numpy as np
import z3

from treeck import *
from treeck.verifier import Verifier, DefaultVerifier
from treeck.z3backend import Z3Backend as Backend

class TestVerifier(unittest.TestCase):
    def test_single_tree(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        v = Verifier([RealDomain()], at, Backend())
        v._initialize()
        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.SAT)
        v._backend.add_constraint(v._enc_tree(0, at[0]))
        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.41), Verifier.Result.UNSAT)

        v = Verifier([RealDomain(1, 3)], at, Backend())
        v.add_constraint(v.xvar(0) < 2.0)
        v._initialize()
        self.assertEqual(v._reachability[(0, 2.0)], Verifier.Reachable.LEFT)
        self.assertEqual(v._reachability[(0, 1.0)], Verifier.Reachable.RIGHT)
        v._backend.add_constraint(v._enc_tree(0, at[0]))
        self.assertEqual(v._backend.check(v.fvar() != 0.2), Verifier.Result.UNSAT)

        # DefaultVerifier
        dv = DefaultVerifier([RealDomain()], at, Backend())
        self.assertEqual(dv.verify(dv.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(dv.verify(dv.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.41), Verifier.Result.UNSAT)
        #print(dv._backend._solver)

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
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 1, 1)
        t.split(t.right(t.root()), 1, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), -0.3)
        t.set_leaf_value(t.right(t.right(t.root())), -0.4)

        dv = DefaultVerifier([RealDomain(), RealDomain()], at, Backend())
        self.assertEqual(dv.verify(dv.fvar() < -0.11), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.3, -0.4])
        self.assertEqual(dv.verify(dv.fvar() > -0.09), Verifier.Result.SAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.41), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.39), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.2, 0.2])

        dv = DefaultVerifier([RealDomain(), RealDomain()], at, Backend())
        dv.add_constraint(dv.xvar(0) < 2.0)
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.39), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.2, 0.2])

        dv = DefaultVerifier([RealDomain(), RealDomain()], at, Backend())
        dv.add_constraint(dv.xvar(0) >= 2.0)
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.3, -0.4])
        self.assertEqual(dv.verify(dv.fvar() > 0.39), Verifier.Result.UNSAT)


        doms = [RealDomain(), RealDomain()]; doms[0].hi = 2.0
        dv = DefaultVerifier(doms, at, Backend())
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() > 0.39), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.2, 0.2])

        doms = [RealDomain(), RealDomain()]; doms[0].lo = 2.0
        dv = DefaultVerifier(doms, at, Backend())
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.SAT)
        self.assertEqual(dv.model()["ws"], [0.3, -0.4])
        self.assertEqual(dv.verify(dv.fvar() > 0.39), Verifier.Result.UNSAT)
        dv.add_constraint(dv.xvar(1) < 2.0)
        self.assertEqual(dv.verify(dv.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(dv.verify(dv.fvar() < 0.01), Verifier.Result.SAT)

        model = dv.model()
        self.assertEqual(model["ws"], [0.3, -0.3])
        dv.exclude_model(model)
        self.assertEqual(dv.verify(dv.fvar() < 0.01, reset=False), Verifier.Result.SAT)
        model = dv.model()
        self.assertEqual(model["ws"], [0.3, -0.3])
        dv.exclude_model(model)
        self.assertEqual(dv.verify(dv.fvar() < 0.01, reset=False), Verifier.Result.UNSAT)

    def test_img(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")

        m, M = min(ys), max(ys)
        img = np.array(ys).reshape((100, 100))

        #fig, ax = plt.subplots(2, 2)
        #ax[0, 0].imshow(img0, vmin=m, vmax=M)
        #ax[0, 1].imshow(img1, vmin=m, vmax=M)
        #ax[1, 0].imshow(img2, vmin=m, vmax=M)
        #ax[1, 1].imshow(img3, vmin=m, vmax=M)
        #plt.show()

        print("< 0")
        dv = DefaultVerifier([RealDomain(), RealDomain()], at, Backend())
        self.assertEqual(dv.verify(dv.fvar() < 0.0), Verifier.Result.SAT)
        model = dv.model()
        self.assertLess(model["f"], 0.0)
        self.assertGreaterEqual(model["f"], m)

        print("< m, > M")
        self.assertEqual(dv.verify((dv.fvar() < m) | (dv.fvar() > M)), Verifier.Result.UNSAT)

        quandrant = 0
        img = np.array(ys).reshape((100, 100))
        for x0 in [0, 50]:
            for y0 in [0, 50]:
                print("quadrant", quandrant)
                x1, y1 = x0 + 50, y0 + 50
                imgq = img[x0:x1, y0:y1]
                m, M = imgq.min(), imgq.max()

                dv = DefaultVerifier([RealDomain(x0, x1), RealDomain(y0, y1)], at, Backend())
                self.assertEqual(dv.verify(dv.fvar() < m+1e-4), Verifier.Result.SAT)
                self.assertAlmostEqual(dv.model()["f"], m, delta=1e-4)
                self.assertEqual(dv.verify(dv.fvar() > M-1e-4), Verifier.Result.SAT)
                self.assertAlmostEqual(dv.model()["f"], M, delta=1e-4)

                quandrant += 1

    def test_img_sampling(self):
        # find all points with predictions less than 0.0
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        img = np.array(ys).reshape((100, 100))
        at = AddTree.read("tests/models/xgb-img-easy.json")
        dv = DefaultVerifier([RealDomain(), RealDomain()], at, Backend())
        dv.add_constraint(dv.fvar() < 0.0)

        models = []
        while dv.verify(reset=False) == Verifier.Result.SAT:
            m = dv.model()
            x = int(np.floor(m["xs"][1]))
            y = int(np.floor(m["xs"][0]))
            self.assertLess(m["f"], 0.0)
            self.assertAlmostEqual(img[y][x], m["f"], delta=1e-4)
            models.append((x, y))
            dv.exclude_model(m)

        self.assertEqual(len(models), 60)

        #fig, ax = plt.subplots()
        #ax.imshow(img)
        #for (x, y) in models:
        #    ax.scatter([x], [y], marker="s", c="b")
        #plt.show()

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
