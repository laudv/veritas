# Copyright 2019 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import math, timeit
from bisect import bisect

from enum import Enum
from . import AddTree, Interval

class VerifierExpr:
    pass

class VerifierRealExpr(VerifierExpr):
    pass

class VerifierBoolExpr(VerifierExpr):
    def __and__(self, other):
        return VerifierAndExpr(self, other)

    def __or__(self, other):
        return VerifierOrExpr(self, other)

class VerifierVar:
    def __init__(self, verifier):
        self._verifier = verifier

    def get(self):
        raise RuntimeError("abstract method")

class Xvar(VerifierVar, VerifierRealExpr, VerifierBoolExpr): 
    def __init__(self, verifier, feat_id):
        super().__init__(verifier)
        self._feat_id = feat_id

    def get(self):
        return self._verifier._xvars[self._feat_id]

class Rvar(VerifierVar, VerifierRealExpr): # An additional real variable
    def __init__(self, verifier, name):
        super().__init__(verifier)
        self._name = name

    def get(self):
        return self._verifier._rvars[self._name]

class Wvar(VerifierVar, VerifierRealExpr):
    def __init__(self, verifier, tree_index):
        super().__init__(verifier)
        self._tree_index = tree_index

    def get(self):
        return self._verifier._wvars[self._tree_index]

class Fvar(VerifierVar, VerifierRealExpr):
    def __init__(self, verifier):
        super().__init__(verifier)

    def get(self):
        return self._verifier._fvar

class SumExpr(VerifierRealExpr):
    def __init__(self, *parts):
        """
        Sum up expressions `parts`. The parts are VerifierExpr,
        VerifierVar, or floats.
        """
        assert len(parts) > 0
        self.parts = parts

ORDER_CONSTRAINTS = [
    ("VerifierLtExpr", "__lt__"),
    ("VerifierGtExpr", "__gt__"),
    ("VerifierLeExpr", "__le__"),
    ("VerifierGeExpr", "__ge__"),
    ("VerifierEqExpr", "__eq__"),
    ("VerifierNeExpr", "__ne__")]

for (clazz, method) in ORDER_CONSTRAINTS:
    exec(f"""
class {clazz}(VerifierBoolExpr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
""")
    locs = {"f": None}
    exec(f"""
def f(self, other):
    return {clazz}(self, other)
""", globals(), locs)
    setattr(VerifierRealExpr, method, locs["f"])

class VerifierAndExpr(VerifierBoolExpr):
    def __init__(self, *conjuncts):
        self.conjuncts = []
        for c in conjuncts:
            if isinstance(c, VerifierAndExpr): self.conjuncts += c.conjuncts;
            else: self.conjuncts.append(c)

class VerifierOrExpr(VerifierBoolExpr):
    def __init__(self, *disjuncts):
        self.disjuncts = []
        for d in disjuncts:
            if isinstance(d, VerifierOrExpr): self.disjuncts += d.disjuncts;
            else: self.disjuncts.append(d)

class VerifierNotExpr(VerifierBoolExpr):
    def __init__(self, expr):
        self.expr = expr


def encode_prune_box(verifier, prune_box):
    cs = []
    for feat_id, interval in prune_box.items():
        if interval.is_everything():
            raise RuntimeError("Unconstrained feature -> should not be in dict")
        var = verifier.xvar(feat_id)
        if interval.lo_is_unbound(): cs.append(var <  interval.hi)
        elif interval.hi_is_unbound(): cs.append(var >= interval.lo)
        else:
            cs.append((var >= interval.lo) & (var < interval.hi))
    return VerifierAndExpr(*cs)


# -----------------------------------------------------------------------------


class VerifierBackend:

    def set_timeout(self, timeout):
        """
        Set the maximum number of SECONDS the backend is allowed to spend.
        Return UNKNOWN for check tasks if they take longer.
        """
        raise RuntimeError("abstract method")

    def add_real_var(self, name):
        """ Add a new real variable to the session. """
        raise RuntimeError("abstract method")

    def add_bool_var(self, name):
        """ Add a new boolean variable to the session. """
        raise RuntimeError("abstract method")

    def add_constraint(self, constraint):
        """
        Add a constraint to the current session. Constraint can be a
        VerifierBoolExpr or a Backend specific constraint.
        """
        raise RuntimeError("abstract method")

    def simplify(self):
        """ Given the backend a chance to process a bunch of added constraints. """
        raise RuntimeError("abstract method")

    def encode_leaf(self, tree_var, leaf_value):
        """ Encode the leaf node """
        raise RuntimeError("abstract method")

    def encode_internal(self, split, left, right):
        """
        Encode an internal node splitting on `split` and branches `left` and
        `right`.
        """
        raise RuntimeError("abstract method")

    def encode_split(self, feat_var, split):
        """ Encode the given split test. """
        raise RuntimeError("abstract method")

    def check(self, *constraints):
        """ Satisfiability check, optionally with additional constraints. """
        raise RuntimeError("abstract method")

    def model(self, *name_vars_pairs):
        """
        Get assignment to the given variables. The format of name_vars_pairs is:
            `(name1, [var1, var2, ...]), (name2, var), ...`
        Returns a dictionary:
            { name1: [var1_value, var2_value, ...], name2: var_value, ... }
        """
        raise RuntimeError("abstract method")



# -----------------------------------------------------------------------------



class VerifierTimeout(Exception):
    def __init__(self, unk_after):
        msg = "Backend Timeout: UNKNOWN returned after {:.3f} seconds".format(unk_after)
        super().__init__(msg)
        self.unk_after = unk_after



# -----------------------------------------------------------------------------



class Verifier:
    class Result(Enum):
        SAT = 1
        UNSAT = 0
        UNKNOWN = -1

        def is_sat(self):
            if self == Verifier.Result.UNKNOWN:
                raise RuntimeError(f"Unexpected {Verifier.Result.UNKNOWN}")
            return self == Verifier.Result.SAT

        def __str__(self):
            if self == Verifier.Result.SAT:     return "SAT"
            if self == Verifier.Result.UNSAT:   return "UNSAT"
            if self == Verifier.Result.UNKNOWN: return "UNKNOWN"

    def __init__(self, addtree, prune_box, backend):
        assert isinstance(addtree, AddTree)
        assert isinstance(prune_box, dict)
        if len(prune_box.keys())>0:
            for k, v in prune_box.items():
                assert isinstance(k, int)
                assert isinstance(v, Interval)
        assert isinstance(backend, VerifierBackend)

        # 3 blocks: verifier backend, addtree, prune_box
        self._backend = backend
        self._addtree = addtree
        self._prune_box = prune_box

        # initialize
        self._status = Verifier.Result.UNKNOWN
        self.check_time = -math.inf
        self.nchecks = 0

        # extract variables from addtree
        self._xvars = {fid: self._backend.add_real_var(f"x{fid}")
                for fid in range(self._addtree.get_maximum_feat_id()+1)}
        self._wvars = [self._backend.add_real_var(f"w{i}")
                for i in range(len(self._addtree))]
        self._fvar = self._backend.add_real_var(f"f")

        # possibily, additional real variables
        self._rvars = {} 

        ### NOTE what if multi-class? I manually put a "0" into get_base_score now
        # FVAR = sum{WVARS}
        fexpr = SumExpr(self._addtree.get_base_score(0), *self._wvars)
        self.add_constraint(fexpr == self.fvar())

        # add constraints from prune_box 
        if len(self._prune_box.keys())>0:
            # (!) add xvars not present in addtree (rare, but possible)
            # --> if addtree uses X1, X2 but *not* X3, we don't have X3 in model
            for fid in self._prune_box.keys():
                if fid not in self._xvars.keys():
                    self._xvars[fid] = self._backend.add_real_var(f"x{fid}")
            self.add_constraint(encode_prune_box(self, self._prune_box))


        #print(self._xvars, self._wvars, self._fvar, self._backend._solver)

        self._splits = None
        self.leaf_count = 0


    def xvar(self, feat_id):
        """ Get the decision variable associated with feature `feat_id`. """
        return Xvar(self, feat_id)

    def wvar(self, tree_index):
        """ Get the decision variable associated with the output value of tree `tree_index`. """
        return Wvar(self, tree_index)

    def fvar(self):
        """ Get the decision variable associated with the output of the model. """
        return Fvar(self)

    def add_rvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._rvars
        rvar = self._backend.add_real_var(f"r_{name}")
        self._rvars[name] = rvar

    def rvar(self, name):
        """ Get one of the additional decision variables. """
        return Rvar(self, name)


    def add_all_trees(self):
        """ Add all trees in the addtree. """
        for tree_index in range(len(self._addtree)):
            self.add_tree(tree_index)

    def add_tree(self, tree_index):
        """ Add the full encoding of a tree to the backend.  """
        tree = self._addtree[tree_index]
        enc = self._enc_tree(tree, tree.root(), tree_index)
        self._backend.add_constraint(enc)
        lo, hi = tree.find_minmax_leaf_value(tree.root())[0] 
        wvar = self._wvars[tree_index]
        if not math.isinf(lo):
            self._backend.add_constraint(wvar >= lo)
        if not math.isinf(hi):
            self._backend.add_constraint(wvar <= hi)

    def _enc_tree(self, tree, node, tree_index):
        if tree.is_leaf(node):
            wvar = self._wvars[tree_index]
            leaf_value = tree.get_leaf_value(node, 0) ### CLASS 0 ONLY???
            self.leaf_count += 1
            return self._backend.encode_leaf(wvar, leaf_value)
        else:
            split = tree.get_split(node)
            xvar = self._xvars[split.feat_id]
            left, right = tree.left(node), tree.right(node)
            l = self._enc_tree(tree, left, tree_index)
            r = self._enc_tree(tree, right, tree_index)
            split_enc = self._backend.encode_split(xvar, split)
            return self._backend.encode_internal(split_enc, l, r)


    def add_constraint(self, constraint):
        """
        Add a user-defined constraint. Use add_rvar, rvar, bvar, xvar, and fvar
        to get access to the variables.
        """
        return self._backend.add_constraint(constraint)

    def set_timeout(self, timeout):
        """ Set the timeout of the backend solver. """
        self._backend.set_timeout(timeout)


    def check(self, *constraints):
        """
        Throws VerifierTimeout when output of backend is UNKNOWN, so output is
        ensured to be SAT or UNSAT.
        """
        t0 = timeit.default_timer()
        status = self._backend.check(*constraints)
        self.nchecks += 1
        t1 = timeit.default_timer()
        self.check_time = t1 - t0

        if status == Verifier.Result.UNKNOWN:
            raise VerifierTimeout(self.check_time)
        return status

    def model(self):
        """
        If a `check` was successful, i.e., the output was
        `Verifier.Result.SAT`, then this method returns a `dict` containing the
        variable assignments that made the model SAT. The dict has the
        following structure:

        dict{
            "xs": [ list of xvar values ],
            "ws": [ list of tree leaf weights ],
            "f": sum of tree leaf weights == addtree.base_score + sum{ws}
            "rs": { name => value } value map of additional real variables
            "bs": { name => value } value map of additional bool variables
            }
        """
        args = [("xs", self._xvars),
                ("ws", self._wvars),
                ("f",  self._fvar )]

        args.append(("rs", self._rvars))
        
        return self._backend.model(*args)

    def model_family(self, model):
        """
        Get ranges on the xvar values within which the model does not
        change its predicted value.
        """
        xs = model["xs"]
        
        if isinstance(xs, dict):
            xs = list(xs.values())

        # create list of reached leaves (select [0] as otherwise it's list of lists)
        leafs = [tree.eval_node(xs)[0] for tree in self._addtree]

        return self._addtree.compute_box(leafs)


    

