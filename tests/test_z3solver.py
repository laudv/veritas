import matplotlib.pyplot as plt
import unittest, json
import numpy as np
import z3

from treeck import *

class TestZ3Solver(unittest.TestCase):
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

        #print(at)

        solver = Z3Solver([RealDomain()], at)
        self.assertEqual(solver.verify(threshold=0, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(threshold=0, op=GREATER_THAN), z3.sat)
        self.assertEqual(solver.verify(threshold=0.41, op=LESS_THAN), z3.sat)
        self.assertEqual(solver.verify(threshold=0.41, op=GREATER_THAN), z3.unsat)

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

        #print(at)

        solver = Z3Solver([RealDomain(), RealDomain()], at)
        self.assertEqual(solver.verify(threshold=-0.11, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(threshold=-0.09, op=LESS_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.3, -0.4])
        self.assertEqual(solver.verify(threshold=-0.09, op=GREATER_THAN), z3.sat)
        self.assertEqual(solver.verify(threshold=0.41, op=GREATER_THAN), z3.unsat)
        self.assertEqual(solver.verify(threshold=0.39, op=GREATER_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.2, 0.2])

        cs = [solver.xvar(0) < 2]
        self.assertEqual(solver.verify(cs, threshold=-0.09, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(cs, threshold=0.39, op=GREATER_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.2, 0.2])

        cs = [solver.xvar(0) >= 2]
        self.assertEqual(solver.verify(cs, threshold=-0.09, op=LESS_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.3, -0.4])
        self.assertEqual(solver.verify(cs, threshold=0.39, op=GREATER_THAN), z3.unsat)

        doms = [RealDomain(), RealDomain()]; doms[0].hi = 2
        solver = Z3Solver(doms, at)
        self.assertEqual(solver.verify(threshold=-0.09, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(threshold=0.39, op=GREATER_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.2, 0.2])

        doms = [RealDomain(), RealDomain()]; doms[0].lo = 2
        solver = Z3Solver(doms, at)
        self.assertEqual(solver.verify(threshold=-0.09, op=LESS_THAN), z3.sat)
        self.assertEqual(solver.model()["ws"], [0.3, -0.4])
        self.assertEqual(solver.verify(threshold=0.39, op=GREATER_THAN), z3.unsat)
        cs = [solver.xvar(1) < 2]
        self.assertEqual(solver.verify(cs, threshold=-0.09, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(cs, threshold= 0.01, op=LESS_THAN), z3.sat)

    def test_img(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")

        img = np.array(ys).reshape((100, 100))
        img0 = img[0:50, 0:50]
        img1 = img[0:50, 50:100]
        img2 = img[50:100, 0:50]
        img3 = img[50:100, 50:100]

        #fig, ax = plt.subplots(2, 2)
        #ax[0, 0].imshow(img0)
        #ax[0, 1].imshow(img1)
        #ax[1, 0].imshow(img2)
        #ax[1, 1].imshow(img3)
        #plt.show()

        m, M = min(ys), max(ys)
        m0, M0 = img0.min(), img0.max()
        m1, M1 = img1.min(), img1.max()
        m2, M2 = img2.min(), img2.max()
        m3, M3 = img3.min(), img3.max()

        print("< 0")
        solver = Z3Solver([RealDomain(), RealDomain()], at)
        self.assertEqual(solver.verify(threshold=0.0, op=LESS_THAN), z3.sat)

        print("< m, > M")
        self.assertEqual(solver.verify(threshold=m, op=LESS_THAN), z3.unsat)
        self.assertEqual(solver.verify(threshold=M, op=GREATER_THAN), z3.unsat)

        print("quadrant 1")
        solver = Z3Solver([RealDomain(0, 50), RealDomain(0, 50)], at)
        self.assertEqual(solver.verify(threshold=m0+1e-4, op=LESS_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, m0, delta=1e-4)
        self.assertEqual(solver.verify(threshold=M0-1e-4, op=GREATER_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, M0, delta=1e-4)

        print("quadrant 2")
        solver = Z3Solver([RealDomain(0, 50), RealDomain(50, 100)], at)
        self.assertEqual(solver.verify(threshold=m1+1e-4, op=LESS_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, m1, delta=1e-4)
        self.assertEqual(solver.verify(threshold=M1-1e-4, op=GREATER_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, M1, delta=1e-4)

        print("quadrant 3")
        solver = Z3Solver([RealDomain(50, 100), RealDomain(0, 50)], at)
        self.assertEqual(solver.verify(threshold=m2+1e-4, op=LESS_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, m2, delta=1e-4)
        self.assertEqual(solver.verify(threshold=M2-1e-4, op=GREATER_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, M2, delta=1e-4)

        print("quadrant 4")
        solver = Z3Solver([RealDomain(50, 100), RealDomain(50, 100)], at)
        self.assertEqual(solver.verify(threshold=m3+1e-4, op=LESS_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, m3, delta=1e-4)
        self.assertEqual(solver.verify(threshold=M3-1e-4, op=GREATER_THAN), z3.sat)
        self.assertAlmostEqual(sum(solver.model()["ws"]) + at.base_score, M3, delta=1e-4)


if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
