import unittest
import z3

from treeck import *
from treeck.verifier import Verifier, LtConstraint, Dvar
from treeck.z3backend import Z3Backend

class DummyVerifier:
    def __init__(self, backend):
        self.b = backend
        self._dvars = {}

    def add_var(self, name):
        v = self.b.add_var(name)
        self._dvars[name] = v
        return v

class TestZ3Solver(unittest.TestCase):
    def test_dummy_verifier_interaction(self):
        b = Z3Backend()
        v = DummyVerifier(b)

        x = Dvar(v, "x")
        y = Dvar(v, "y")
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

    def test_tree(self):
        b = Z3Backend()

        w = b.add_var("w1")
        x = b.add_var("x")

        ll = b.encode_leaf(w, 1.0)
        lr = b.encode_leaf(w, 2.0)
        s = b.encode_split(x, 5.0, ll, lr)
        b.add_constraint(s)

        b.add_constraint((w > 1.5))
        status = b.check()
        self.assertEqual(status, Verifier.Result.SAT)
        x_value = b.model(("x", x))["x"]
        self.assertGreaterEqual(x_value, 5.0)

        b.add_constraint((x < 5.0))
        status = b.check()
        self.assertEqual(status, Verifier.Result.UNSAT)
        self.assertEqual(b.stats()["num_check_calls"], 2)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()