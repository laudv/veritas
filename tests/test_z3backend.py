import unittest
import z3

from veritas import *
from veritas.smt import Verifier, Rvar, SumExpr
from veritas.z3backend import Z3Backend

class DummyVerifier:
    def __init__(self, backend):
        self.b = backend
        self._rvars = {}

    def add_var(self, name):
        v = self.b.add_real_var(name)
        self._rvars[name] = v
        return v

class TestZ3Backend(unittest.TestCase):
    def test_dummy_verifier_interaction(self):
        b = Z3Backend()
        v = DummyVerifier(b)

        x = Rvar(v, "x")
        y = Rvar(v, "y")
        cs = [x < y, x==1.0, y==2.0]

        zx = v.add_var("x")
        zy = v.add_var("y")

        for c in cs:
            b.add_constraint(c)

        status = b.check()
        self.assertEqual(status, Verifier.Result.SAT)
        m = b.model(("all", [zx, zy]), ("x", zx), ("y", zy))
        self.assertEqual(m["all"], [1.0, 2.0])
        self.assertEqual(m["x"], 1.0)
        self.assertEqual(m["y"], 2.0)

        status = b.check(x > y)
        self.assertEqual(status, Verifier.Result.UNSAT)

        status = b.check()
        self.assertEqual(status, Verifier.Result.SAT)

        b.add_constraint(x > y)
        status = b.check()
        self.assertEqual(status, Verifier.Result.UNSAT)

    def test_tree_lt(self):
        b = Z3Backend()

        w = b.add_real_var("w1")
        x = b.add_real_var("x")

        ll = b.encode_leaf(w, 1.0)
        lr = b.encode_leaf(w, 2.0)
        s = b.encode_split(x, LtSplit(0, 5.0))
        tr = b.encode_internal(s, ll, lr)
        b.add_constraint(tr)

        b.add_constraint((w > 1.5))
        status = b.check()
        self.assertEqual(status, Verifier.Result.SAT)
        x_value = b.model(("x", x))["x"]
        self.assertGreaterEqual(x_value, 5.0)

        b.add_constraint((x < 5.0))
        status = b.check()
        self.assertEqual(status, Verifier.Result.UNSAT)


    def test_sum_expr(self):
        b = Z3Backend()

        w1 = b.add_real_var("w1")
        w2 = b.add_real_var("w2")
        w3 = b.add_real_var("w3")
        x = b.add_real_var("x")

        b.add_constraint(w1 < 1)
        b.add_constraint(w2 < 2)
        b.add_constraint(w3 < 3)
        b.add_constraint(x > 15)

        s1 = SumExpr(w1, w2)
        s = SumExpr(s1, 10.0, w3)
        b.add_constraint(s > x)
        status = b.check()
        self.assertEqual(status, Verifier.Result.SAT)

        b.add_constraint(w3 < 1)
        status = b.check()
        self.assertEqual(status, Verifier.Result.UNSAT)


if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
