import z3

from .verifier import Verifier, ORDER_CONSTRAINTS
from .verifier import VerifierBoolExpr, VerifierVar
from .verifier import VerifierLtExpr, VerifierGtExpr, VerifierLeExpr, VerifierGeExpr, VerifierEqExpr, VerifierNeExpr
from .verifier import VerifierAndExpr, VerifierOrExpr
from .verifier import SumExpr
from .verifier import VerifierBackend

class Stats:
    def __init__(self):
        self.num_check_calls = 0
        self.num_simplifies = 0

class Z3Backend(VerifierBackend):

    ORDER_CONSTRAINTS_MAP = dict(
        [(eval(name), method) for (name, method) in ORDER_CONSTRAINTS])

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

    def add_real_var(self, name):
        return z3.Real(name, self._ctx)

    def add_bool_var(self, name):
        return z3.Bool(name, self._ctx)

    def add_constraint(self, *constraints):
        encs = self._enc_constraints(constraints)
        self._solver.add(*encs)

    def simplify(self):
        self._stats.num_simplifies += 1

    def encode_leaf(self, tree_var, leaf_value):
        return (tree_var == leaf_value)

    def encode_split(self, feat_var, split_value, left, right):
        if left == False and right == False:
            return False
        cond = (feat_var < split_value)
        if left == False:
            return z3.And(z3.Not(cond, self._ctx), right, self._ctx)
        if right == False:
            return z3.And(cond, left, self._ctx)
        return z3.If(cond, left, right, self._ctx)

    def check(self, *constraints):
        encs = self._enc_constraints(constraints)
        if isinstance(encs, bool) and not encs:
            return Verifier.Result.UNSAT
        status = self._solver.check(*encs)
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
            elif isinstance(vs, dict):
                model[name] = dict([(k, self._extract_var(z3model, v)) for (k, v) in vs.items()])
            else:
                model[name] = self._extract_var(z3model, vs)
        return model

    # -- private --

    def _enc_constraints(self, cs):
        encs = []
        for c in cs:
            enc = self._enc_constraint(c)
            if isinstance(enc, bool):
                if not enc: return False
                # skip True
            else: encs.append(enc)
        return encs

    def _enc_constraint(self, c):
        return self._enc_bool_expr(c)

    def _enc_bool_expr(self, c):
        if z3.is_bool(c):
            return c
        elif isinstance(c, bool):
            return c
        elif isinstance(c, VerifierBoolExpr):
            return self._enc_verifier_bool_expr(c)
        else:
            raise RuntimeError("unsupported bool expression of type",
                    type(c).__qualname__)

    def _enc_verifier_bool_expr(self, c):
        if isinstance(c, VerifierAndExpr):
            cs = list(map(self._enc_bool_expr, c.conjuncts))
            return z3.And(*cs, self._ctx) if len(cs) > 0 else True
        elif isinstance(c, VerifierOrExpr):
            cs = list(map(self._enc_bool_expr, c.disjuncts))
            return z3.Or(*cs, self._ctx) if len(cs) > 0 else False
        elif type(c) in Z3Backend.ORDER_CONSTRAINTS_MAP.keys():
            real1 = self._enc_real_expr(c.lhs)
            real2 = self._enc_real_expr(c.rhs)
            method = Z3Backend.ORDER_CONSTRAINTS_MAP[type(c)]
            method = getattr(real1, method)
            return method(real2) # call float's or Z3's ArithRef's __lt__, __gt__, ...
        elif isinstance(c, VerifierVar):
            return c.get()
        else:
            raise RuntimeError("unsupported VerifierBoolExpr of type",
                    type(c).__qualname__)

    def _enc_real_expr(self, c):
        if z3.is_real(c):
            return c
        elif isinstance(c, float):
            return c
        elif isinstance(c, int):
            return float(c)
        elif isinstance(c, VerifierVar):
            return c.get()
        elif isinstance(c, SumExpr):
            s = self._enc_real_expr(c.parts[0])
            for p in c.parts[1:]:
                s += self._enc_real_expr(p)
            return s
        else:
            raise RuntimeError("unsupported VerifierRealExpr of type",
                    type(c).__qualname__)

    def _extract_var(self, z3model, var):
        val = z3model[var]
        if val is None:
            return None
        if z3.is_rational_value(val):
            n = val.numerator_as_long()
            d = val.denominator_as_long()
            return float(n / d)
        raise RuntimeError("var not supported")
