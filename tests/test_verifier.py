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
        self.assertEqual(dv.verify(dv.fvar() < 0.01, reset=False), Verifier.Result.UNSAT)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
