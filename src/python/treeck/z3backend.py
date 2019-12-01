import z3

from .verifier import Verifier, FUNDAMENTAL_CONSTRAINTS
from .verifier import VerifierConstraint, CompoundVerifierConstraint, ConstraintVar
from .verifier import LtConstraint, GtConstraint, LeConstraint, GeConstraint, EqConstraint, NeConstraint
from .verifier import SumExpr
from .verifier import VerifierBackend

class Stats:
    def __init__(self):
        self.num_check_calls = 0

class Z3Backend(VerifierBackend):

    FUNDAMENTAL_CONSTRAINTS_MAP = dict(
            [(eval(name), method) for (name, method) in FUNDAMENTAL_CONSTRAINTS])

    def __init__(self):
        self._ctx = z3.Context()
        self.reset()

    def stats(self):
        if self._stats:
            return vars(self._stats)
        else: return {}

    def reset(self):
        self._solver = z3.Solver(ctx=self._ctx)
        self._stats = Stats()

    def add_var(self, name):
        return z3.Real(name, self._ctx)

    def add_constraint(self, constraint):
        for enc in self._enc_constraint(constraint):
            self._solver.add(enc)

    def encode_leaf(self, tree_var, leaf_value):
        return (tree_var == leaf_value)

    def encode_split(self, feat_var, split_value, left, right):
        if left == False and right == False:
            return False
        cond = (feat_var < split_value)
        if left == False:
            return z3.And(z3.Not(cond), right, self._ctx)
        if right == False:
            return z3.And(cond, right, self._ctx)
        return z3.If(cond, left, right, self._ctx)

    def check(self, *constraints):
        encs = []
        for c in constraints:
            for enc in self._enc_constraint(c):
                encs.append(enc)

        status = self._solver.check(encs)
        self._stats.num_check_calls += 1
        if status == z3.sat:     return Verifier.Result.SAT
        elif status == z3.unsat: return Verifier.Result.UNSAT
        else:                    return Verifier.Result.UNKNOWN

    def model(self, *name_vars_pairs):
        model = {}
        z3model = self._solver.model()
        for (name, vs) in name_vars_pairs:
            if isinstance(vs, list):
                model[name] = [self._extract_var(z3model, v) for v in vs]
            else:
                model[name] = self._extract_var(z3model, vs)
        return model

    # -- private --

    def _enc_constraint(self, c):
        if z3.is_bool(c):
            yield c
        elif isinstance(c, bool):
            yield c
        elif isinstance(c, CompoundVerifierConstraint):
            for comp in c.compounds:
                self._enc_constraint(comp)
        elif isinstance(c, VerifierConstraint):
            fmap = Z3Backend.FUNDAMENTAL_CONSTRAINTS_MAP
            tp = type(c)
            if tp in fmap.keys():
                expr1 = self._enc_expr(c.var1)
                expr2 = self._enc_expr(c.var2)
                f = getattr(expr1, fmap[tp]) # call float's or Z3's ArithRef's __lt__, __gt__, ...
                yield f(expr2)
        else: raise RuntimeError("constraint not supported: {} of type {}".format(c, type(c)))

    def _enc_expr(self, e):
        if z3.is_real(e):
            return e
        if isinstance(e, float):
            return e
        if isinstance(e, ConstraintVar):
            return e.get()
        if isinstance(e, SumExpr):
            s = self._enc_expr(e.parts[0])
            for p in e.parts[1:]:
                s += self._enc_expr(p)
            return s
        raise RuntimeError("expr not supported: {} of type {}".format(e, type(e)))

    def _extract_var(self, z3model, var):
        val = z3model[var]
        if val is None:
            return None
        if z3.is_rational_value(val):
            n = val.numerator_as_long()
            d = val.denominator_as_long()
            return float(n / d)
        raise RuntimeError("var not supported")
