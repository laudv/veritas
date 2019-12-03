import math
from bisect import bisect

from enum import Enum

from . import RealDomain


class VerifierExpr:
    pass

class VerifierRealExpr(VerifierExpr):
    pass

class VerifierBoolExpr(VerifierExpr):
    def __and__(self, other):
        if isinstance(self, VerifierAndExpr):
            if isinstance(other, VerifierAndExpr):
                self.conjuncts += other.conjuncts
            else:
                self.conjuncts.append(other)
        else:
            return VerifierAndExpr(self, other)

    def __or__(self, other):
        if isinstance(self, VerifierOrExpr):
            if isinstance(other, VerifierOrExpr):
                self.disjuncts += other.disjuncts
            else:
                self.disjuncts.append(other)
            return self
        else:
            return VerifierOrExpr(self, other)

class VerifierVar(VerifierRealExpr):
    def __init__(self, verifier):
        self._verifier = verifier

    def get(self):
        raise RuntimeError("abstract method")

class Xvar(VerifierVar):
    def __init__(self, verifier, feat_id):
        super().__init__(verifier)
        self._feat_id = feat_id

    def get(self):
        return self._verifier._xvars[self._feat_id]

class Dvar(VerifierVar):
    def __init__(self, verifier, name):
        super().__init__(verifier)
        self._name = name

    def get(self):
        return self._verifier._dvars[self._name]

class Wvar(VerifierVar):
    def __init__(self, verifier, tree_index):
        super().__init__(verifier)
        self._tree_index = tree_index

    def get(self):
        return self._verifier._wvars[self._tree_index]

class Fvar(VerifierVar):
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
        self.conjuncts = conjuncts

class VerifierOrExpr(VerifierBoolExpr):
    def __init__(self, *disjuncts):
        self.disjuncts = disjuncts

class InDomainConstraint(VerifierAndExpr):
    def __init__(self, verifier, domains):
        cs = []
        for feat_id, d in enumerate(domains):
            var = verifier.xvar(feat_id)
            if math.isinf(d.lo) and math.isinf(d.hi):
                continue # unconstrained feature -> everything possible
            elif math.isinf(d.lo): cs.append(var <  d.hi)
            elif math.isinf(d.hi): cs.append(var >= d.lo)
            else:
                cs.append(var >= d.lo)
                cs.append(var <  d.hi)
        super().__init__(*cs)

class NotInDomainConstraint(VerifierOrExpr):
    def __init__(self, verifier, domains):
        cs = []
        for feat_id, d in enumerate(domains):
            var = verifier.xvar(feat_id)
            if math.isinf(d.lo) and math.isinf(d.hi):
                raise RuntimeError("Unconstrained feature -> nothing possible?")
            elif math.isinf(d.lo): cs.append(var >= d.hi)
            elif math.isinf(d.hi): cs.append(var <  d.lo)
            else:
                cs.append(var <  d.lo)
                cs.append(var >= d.hi)
        super().__init__(*cs)

class VerifierBackend:
    def stats(self):
        return {}

    def reset(self):
        """ Terminate the previous session and initialize a new one. """
        raise RuntimeError("abstract method")

    def add_var(self, name):
        """ Add a new variable to the session. """
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

    def encode_split(self, feat_var, split_value, left, right):
        """
        Encode the given split using left and right as the encodings of the
        subtrees.
        """
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



class Verifier:
    class Result(Enum):
        SAT = 1
        UNSAT = 0
        UNKNOWN = -1

        def is_sat(self):
            if self == Verifier.Result.UNKNOWN:
                raise RuntimeError("unhandled Result.UNKNOWN")
            return self == Verifier.Result.SAT

    class Reachable(Enum):
        NONE = 0x0
        LEFT = 0x1
        RIGHT = 0x2
        BOTH = 0x1 | 0x2

        def covers(self, other):
            return self.value & other.value > 0

        def __or__(self, other):
            return Verifier.Reachable(self.value | other.value)

        def __xor__(self, other):
            return Verifier.Reachable(self.value ^ other.value)

    def __init__(self, domains, addtree, backend):
        """
        Initialize a Verifier.
         - domains is a list of `RealDomain` objects, one for each feature.
         - addtree is the model to verify properties of.
        """
        self._constraints = []
        self._domains = domains
        self._addtree = addtree
        self._backend = backend

        self.num_features = len(domains)
        self._xvars = [backend.add_var(f"x{i}") for i in range(self.num_features)]
        self._wvars = [backend.add_var(f"w{i}") for i in range(len(self._addtree))]
        self._dvars = {}
        self._fvar = backend.add_var("f")

        self._splits = None

        # (feat_id, split_value) => REACHABILITY FLAG
        self._reachability = {}

    def add_dvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._dvars
        dvar = self._backend.add_var(name)
        self._dvars[name] = dvar

    def dvar(self, name):
        """ Get one of the additional decision variables. """
        return Dvar(self, name)

    def xvar(self, feat_id):
        """ Get the decision variable associated with feature `feat_id`. """
        return Xvar(self, feat_id)

    def wvar(self, tree_index):
        """ Get the decision variable associated with the output value of tree `tree_index`. """
        return Wvar(self, tree_index)

    def fvar(self):
        """ Get the decision variable associated with the output of the model. """
        return Fvar(self)

    def add_constraint(self, constraint):
        """
        Add a user-defined constraint. Use add_dvar, dvar, xvar, and fvar to
        get access to the variables.
        """
        self._constraints.append(constraint)

    def verify(self, constraint=True, timeout=3600 * 24 * 31, reset=True):
        """
        Verify the model, i.e., try to find an assignment to the decision variables that
            (1) satisfies the constraints on
                - the input features (xvars)
                - the addtree output (fvar)
                - any additional decision variables (dvars)
            (2) satisfies the additive tree structure
            (3) satisfies the given constraint

        There are three possible outcomes:
            (1) Verifier.SAT, an assignment was found
            (2) Verifier.UNSAT, no assignment that satisfies the constraints possible
            (3) Verifier.UNKNOWN, the answer is unknown, e.g. because of timeout
        """
        raise RuntimeError("abstract method; use subclass")

    def reset(self):
        """
        Reset the verifier so the next call to verify is clean.
        """
        raise RuntimeError("abstract method; use subclass")

    def model(self):
        """
        If a call to `verify` was successful, i.e., the output was
        `Verifier.SAT`, then this method returns a `dict` containing the
        variable assignments that made the model SAT. The dict has the
        following structure:

        dict{
            "xs": [ list of xvar values ],
            "ws": [ list of tree leaf weights ],
            "f": sum of tree leaf weights == addtree.base_score + sum{ws}
            "ds": { name => value } value map of additional variables
            }
        """
        return self._backend.model(
                ("xs", self._xvars),
                ("ws", self._wvars),
                ("f", self._fvar),
                ("ds", self._dvars))

    def exclude_model(self, model):
        """
        Mark the domain region inhabited by `model[xs]` as impossible.

        Usage for model sampling:
        ```
        while cond:
            status = verifier.verify(reset=False)
            if status != Verifier.SAT: break
            model = verifier.model()
            # DO SOMETHING WITH model
            verifier.exclude_model(model)
        ```
        """
        if self._splits is None:
            self._splits = self._addtree.get_splits()

        domains = []
        for i, _, lo, hi in self._find_sample_intervals(model):
            d = RealDomain(lo, hi)
            if d.is_everything():
                raise RuntimeError("Unconstrained feature!")
            #print("{:.6g} <= {:.6g} < {:.6g}".format(lo, x, hi))
            domains.append(d)
        self._backend.add_constraint(NotInDomainConstraint(self, domains))

    def _find_sample_intervals(self, model): # helper `exclude_assignment`
        for i, x in enumerate(model["xs"]):
            if x == None: continue
            split_values = self._splits[i]
            j = bisect(split_values, x)
            lo = -math.inf if j == 0 else split_values[j-1]
            hi = math.inf if j == len(split_values) else split_values[j]
            assert lo < hi
            assert x >= lo
            assert x < hi

            yield i, x, lo, hi

    def get_reachability(self, feat_id, split_value):
        """ Check the reachability of the given split. """
        p = (feat_id, split_value)
        if p in self._reachability:
            return self._reachability[p]
        return Verifier.Reachable.BOTH

    def get_reachability_dict(self):
        """ Get the full unreachable dictionary for reuse in deeper verifiers. """
        return self._reachability.copy()

    def set_reachability_dict(self, reachability):
        self._reachability = reachability

    def _test_addtree_reachability(self):
        """
        For each tree in the addtree, for each internal node in the tree,
        check which side of the split is reachable given the constraints, not
        considering the addtree.

        For solvers, implement `test_split_reachability`.
        """
        for tree_index in range(len(self._addtree)):
            tree = self._addtree[tree_index]
            self._test_tree_reachability(tree)

    def _test_tree_reachability(self, tree):
        """ Test the reachability of the nodes in a single tree.  """
        stack = [(tree.root())]
        while len(stack) > 0:
            node = stack.pop()

            if tree.is_leaf(node): continue

            feat_id, split_value = tree.get_split(node)
            reachability = self.get_reachability(feat_id, split_value)
            xvar = self.xvar(feat_id)

            if reachability.covers(Verifier.Reachable.LEFT):
                check = self._backend.check(xvar < split_value)
                if not check.is_sat():
                    reachability ^= Verifier.Reachable.LEFT # disable left
                else: stack.append(tree.left(node))
            if reachability == Verifier.Reachable.BOTH:          # if left is unreachable, then no ...
                check = self._backend.check(xvar >= split_value) # ... need to test, right is reachable!
                if not check.is_sat():
                    reachability ^= Verifier.Reachable.RIGHT # disable right
                else: stack.append(tree.right(node))

            self._reachability[(feat_id, split_value)] = reachability


    def _initialize(self):
        # - define f as sum of ws
        # - add domain constraints
        # - add user defined constraints
        # - compute reachabilities
        self._backend.reset()
        fexpr = SumExpr(self._addtree.base_score, *self._wvars)
        self._backend.add_constraint(fexpr == self._fvar)
        self._backend.add_constraint(InDomainConstraint(self, self._domains))
        for c in self._constraints:
            self._backend.add_constraint(c)
        self._test_addtree_reachability()
        self._backend.simplify()

    def _enc_tree(self, tree_index, tree, node = None):
        # - start at root
        # - if left/right reachable, recur -> get_reachability
        # - if leaf reached, use backend to encode leaf
        # - use backend encode_split to join branches
        # - add constraint to backend with backend.add_constraint
        if node is None:
            return self._enc_tree(tree_index, tree, tree.root())

        if tree.is_leaf(node):
            wvar = self._wvars[tree_index]
            leaf_value = tree.get_leaf_value(node)
            return self._backend.encode_leaf(wvar, leaf_value)
        else:
            feat_id, split_value = tree.get_split(node)
            xvar = self._xvars[feat_id]
            left, right = tree.left(node), tree.right(node)
            reachability = self.get_reachability(feat_id, split_value)
            l, r = False, False
            if reachability.covers(Verifier.Reachable.LEFT):
                l = self._enc_tree(tree_index, tree, left)
            if reachability.covers(Verifier.Reachable.RIGHT):
                r = self._enc_tree(tree_index, tree, right)
            return self._backend.encode_split(xvar, split_value, l, r)


class DefaultVerifier(Verifier):

    def __init__(self, domains, addtree, backend):
        super().__init__(domains, addtree, backend)
        self._status = Verifier.Result.UNKNOWN

    def verify(self, constraint=True, timeout=3600 * 24 * 31, reset=True):
        """
        DefaultVerifier algorithm:
         - define f as sum of ws          |
         - add domain constraints         |  -> helper method in Verifier, e.g. initialize()
         - add user defined constraints   |
         - compute reachabilities         |
         - add bounds for trees
                -> get_reachability
         - add full encodings of trees in order of best bounds -> get_reachability + backend
                -> helper method enc_tree
         - stop early when UNSAT
         - stop with SAT if SAT with all trees fully encoded
         - stop with UNKNOWN if backend times out
        """

        if reset or self._status == Verifier.Result.UNKNOWN:
            self.reset()
        self._status = self._backend.check(constraint)

        while len(self._remaining_trees) > 0 \
                and self._status == Verifier.Result.SAT:
            self._iteration_count += 1

            # TODO choose best tree according to heuristic
            tree_index = self._remaining_trees.pop()
            tree = self._addtree[tree_index]
            enc = self._enc_tree(tree_index, tree, tree.root())
            self._backend.add_constraint(enc)

            self._status = self._backend.check(constraint)

        return self._status

    def reset(self):
        m = len(self._addtree)
        self._iteration_count = 0
        self._remaining_trees = list(range(m))

        self._initialize() # unreachable available now
        self._bounds = [self._determine_bounds(i) for i in range(m)]
        self._add_bounds_constraints()

        # TODO let user choose heuristic
        # sort remaining trees by heuristic value
        bnd = lambda i: self._bounds[i]
        self._remaining_trees.sort(key=lambda i: max(abs(bnd(i)[0]), abs(bnd(i)[1])))


        self._status = Verifier.Result.SAT

    # -- private --

    def _determine_bounds(self, tree_index):
        tree = self._addtree[tree_index]
        lo =  math.inf
        hi = -math.inf

        stack = [tree.root()]
        while len(stack) > 0:
            node = stack.pop()

            if tree.is_leaf(node):
                leaf_value = tree.get_leaf_value(node)
                lo = min(lo, leaf_value)
                hi = max(hi, leaf_value)
                continue

            feat_id, split_value = tree.get_split(node)
            reachability = self.get_reachability(feat_id, split_value)
            if reachability.covers(Verifier.Reachable.RIGHT):
                stack.append(tree.right(node))
            if reachability.covers(Verifier.Reachable.LEFT):
                stack.append(tree.left(node))

        return (lo, hi)

    def _add_bounds_constraints(self):
        for tree_index, (lo, hi) in enumerate(self._bounds):
            wvar = self.wvar(tree_index)
            self._backend.add_constraint(wvar >= lo)
            self._backend.add_constraint(wvar <= hi)
