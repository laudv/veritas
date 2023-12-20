# Copyright 2019 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import z3

from . import LtSplit#, BoolSplit

from .smt import Verifier, ORDER_CONSTRAINTS
from .smt import VerifierBoolExpr, VerifierVar
from .smt import VerifierLtExpr, VerifierGtExpr, VerifierLeExpr, VerifierGeExpr, VerifierEqExpr, VerifierNeExpr
from .smt import VerifierAndExpr, VerifierOrExpr, VerifierNotExpr
from .smt import SumExpr
from .smt import VerifierBackend

class Z3Backend(VerifierBackend):

    ORDER_CONSTRAINTS_MAP = dict(
        [(eval(name), method) for (name, method) in ORDER_CONSTRAINTS])

    def __init__(self):
        self._ctx = z3.Context()
        self._solver = z3.Solver(ctx=self._ctx)

    def set_timeout(self, timeout):
        self._solver.set("timeout", int(timeout * 1000)) # Z3 seems to interpret timeout as milli seconds

    def add_real_var(self, name):
        return z3.Real(name, self._ctx)

    def add_bool_var(self, name):
        return z3.Bool(name, self._ctx)

    def add_constraint(self, *constraints):
        encs = self._enc_constraints(constraints)
        self._solver.add(*encs)
        return encs

    def simplify(self):
        pass

    def encode_leaf(self, tree_var, leaf_value):
        return (tree_var == leaf_value)

    def encode_internal(self, cond, left, right):
        if left == False and right == False:
            return False
        if left == False:
            return z3.And(z3.Not(cond, self._ctx), right, self._ctx)
        if right == False:
            return z3.And(cond, left, self._ctx)
        return z3.If(cond, left, right, self._ctx)

    def encode_split(self, feat_var, split):
        if isinstance(split, LtSplit):
            return (feat_var < split.split_value) # consistent with AddTree definition
        #elif isinstance(split, BoolSplit):
        #    return feat_var # true goes left, false goes right
        else:
            raise RuntimeError(f"unknown split {split}")

    def check(self, *constraints):
        encs = self._enc_constraints(constraints)
        if isinstance(encs, bool) and not encs:
            return Verifier.Result.UNSAT
        status = self._solver.check(*encs)
        if status == z3.sat:     return Verifier.Result.SAT
        elif status == z3.unsat: return Verifier.Result.UNSAT
        else:                    return Verifier.Result.UNKNOWN

    def model(self, *name_vars_pairs):
        z3model = self._solver.model()
        return self._model_aux(z3model, name_vars_pairs)

    def _model_aux(self, z3model, name_vars_pairs):
        model = {}
        #print("*** call ***")
        #print(name_vars_pairs)
        for (name, vs) in name_vars_pairs:
            #print()
            #print(name, " - ", vs)
            # either (pair (name=>(dict|list|var), list=[var...] or [name=>(dict|list|var)... recur])
            if isinstance(vs, list):
                if len(vs) > 0 and isinstance(vs[0], tuple):
                    #print("recur list:", name, "=>", vs)
                    model[name] = self._model_aux(z3model, vs)
                else:
                    #print("plain list:", name, "=>", vs)
                    model[name] = [self._extract_var(z3model, v) for v in vs]
            elif isinstance(vs, dict):
                #print("recur dict:", name, "=>", vs)
                model[name] = self._model_aux(z3model, list(vs.items()))
            else:
                #print("plain var:", name, "=>", vs)
                model[name] = self._extract_var(z3model, vs)
            #print("model: ", model)
        return model

    # -- private --

    def _enc_constraints(self, cs):
        encs = []
        for c in cs:
            enc = self._enc_constraint(c)
            if isinstance(enc, bool):
                if not enc: return [False]
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
            raise RuntimeError("unsupported expression of type "
                    + type(c).__qualname__)

    def _enc_verifier_bool_expr(self, c):
        
        if isinstance(c, VerifierAndExpr):
            cs = list(map(self._enc_bool_expr, c.conjuncts))
            return z3.And(*cs, self._ctx) if len(cs) > 0 else True
        elif isinstance(c, VerifierOrExpr):
            cs = list(map(self._enc_bool_expr, c.disjuncts))
            return z3.Or(*cs, self._ctx) if len(cs) > 0 else False
        elif isinstance(c, VerifierNotExpr):
            return z3.Not(self._enc_bool_expr(c.expr))
        elif type(c) in Z3Backend.ORDER_CONSTRAINTS_MAP.keys():
            real1 = self._enc_real_expr(c.lhs)
            real2 = self._enc_real_expr(c.rhs)
            method = Z3Backend.ORDER_CONSTRAINTS_MAP[type(c)]
            method = getattr(real1, method)
            return method(real2) # call float's or Z3's ArithRef's __lt__, __gt__, ...
        elif isinstance(c, VerifierVar):
            return c.get()
        else:
            raise RuntimeError("unsupported VerifierBoolExpr of type "
                    + type(c).__qualname__)

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
            raise RuntimeError("unsupported VerifierRealExpr of type "
                    + type(c).__qualname__)

    def _extract_var(self, z3model, var):
        val = z3model[var]
        if val is None:
            return None
        if z3.is_rational_value(val):
            n = val.numerator_as_long()
            d = val.denominator_as_long()
            return float(n / d)
        if z3.is_true(val):
            return True
        if z3.is_false(val):
            return False
        raise RuntimeError("var not supported")
