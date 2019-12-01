import math
import z3

from .verifier import Verifier

class Stats:
    def __init__():
        self.num_check_calls = 0

# TODO remove
class Z3Verifier(Verifier):

    def __init__(self, constraints, domains, addtree):
        super().__init__(constraints, domains, addtree)

        self._num_features = len(domains)
        self._ctx = z3.Context()
        
        self._xvars = [z3.Real(f"x{i}", self._ctx) for i in range(self._num_features)]
        self._wvars = [z3.Real(f"w{i}", self._ctx) for i in range(len(self._addtree))]
        self._dvars = {}
        self._fvar = z3.Real("f", self._ctx)

        self._stats = Stats()

    def add_dvar(self, name):
        dvar = z3.Real(f"d_{name}", self._ctx)
        self._dvars[name] = dvar
        return dvar

    def dvar(self, name):
        return self._dvars[name]

    def xvar(self, feat_id):
        return self._xvars[feat_id]

    def fvar(self):
        return self._fvar

    def verify(self, timeout=3600 * 24 * 31):
        self._remaining_trees = list(range(len(self._addtree)))
        self._iter_count = 0

        self._solver = z3.Solver(ctx=self._ctx)
        self._solver.set("timeout", timeout)
        self._solver.add(self._enc_f_constraint())
        self._solver.add(*self._enc_domain_constraints())
        self._solver.add(*[self._enc_constraint(c) for c in self._constraints])

        self.test_addtree_reachability()

        return Verifier.Result.SAT

    def get_assignment(self):
        z3model = self._solver.model()
        xs = []
        for x in self._xvars:
            xs.append(self._extract_var(z3model, x))
        ws = []
        for w in self._wvars:
            val = self._extract_var(z3model, w)
            assert val is not None
            ws.append(val)
        ds = {}
        for name, dvar in self._dvars.items():
            ds[name] = self._extract_var(z3model, dvar)
        f = self._extract_var(z3model, self._fvar)
        return { "xs": xs, "ws": ws, "ds": ds, "f": f }

    def test_split_reachability(self, feat_id, split_value):
        print("test_split_reachability:", feat_id, split_value)
        xvar = self._xvars[feat_id]
        #status_left = self._check(xvar < split_value)
        #status_right = self._check(
        return Verifier.Reachable.BOTH

    # -- private --

    def _enc_f_constraint(self):
        sumw = self._addtree.base_score
        for w in self._wvars: sumw += w
        return (self._fvar == sumw)

    def _enc_domain_constraints(self):
        cs = []
        for feat_id, dom in enumerate(self._domains):
            if dom.is_everything(): continue

            x = self._xvars[feat_id]

            if not math.isinf(dom.lo):
                if not math.isinf(dom.hi):
                    cs.append(z3.And((dom.lo <= x), (x < dom.hi)))
                else:
                    cs.append((dom.lo <= x))
            else:
                cs.append((x < dom.hi))
        return cs

    def _enc_constraint(self, constraint):
        pass

    def _extract_var(self, z3model, var):
        val = z3model[var]
        if val is None:
            return None
        if z3.is_rational_value(val):
            n = val.numerator_as_long()
            d = val.denominator_as_long()
            return float(n / d)
        raise RuntimeError("var not supported")
    
    def _check(self, constraint = None):
        if constraint:
            status = self._solver.check(constraint)
        else:
            status = self._solver.check()

        self._stats.num_check_calls += 1

        if status == z3.sat:
            return Verifier.Result.SAT
        elif status == z3.unsat:
            return Verifier.Result.UNSAT
        else:
            return Verifier.Result.UNKNOWN
