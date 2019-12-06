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

class VerifierVar:
    def __init__(self, verifier):
        self._verifier = verifier

    def get(self):
        raise RuntimeError("abstract method")

class Xvar(VerifierVar, VerifierRealExpr):
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

class Bvar(VerifierVar, VerifierBoolExpr): # An additional bool variable
    def __init__(self, verifier, name):
        super().__init__(verifier)
        self._name = name

    def get(self):
        return self._verifier._bvars[self._name]

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


# -----------------------------------------------------------------------------


class VerifierBackend:
    def stats(self):
        return {}

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



# -----------------------------------------------------------------------------


class VerifierStrategy:

    def strategy_setup(self, verifier):
        """
        Setup circular dependency between Strategy and Verifier. Verifier is
        responsible for calling this in its constructor.
        """
        pass

    def get_reachability(self, feat_id, split_value):
        """ Given the split, can we go left, right, or both?  """
        return Verifier.Reachable.BOTH

    def verify_setup(self):
        """
        Before starting the verification loop over the verification steps, this
        method is called by the verifier.
        """
        pass

    def verify_step(self):
        """
        This adds the next batch of constraints to the backend. Returns True if
        there is new work to verify, or False if finished.
        """
        raise RuntimeError("abstract method")

    def verify_teardown(self):
        """
        Called after the verification loop, regardless of outcome. Check
        outcome in `verifier._status`.
        """
        pass


# -----------------------------------------------------------------------------



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


    def __init__(self, domains, addtree, backend, strategy, prefix=""):
        """
        Initialize a Verifier.
         - domains is a list of `RealDomain` objects, one for each feature.
         - addtree is the model to verify properties of.
        """
        self._domains = domains
        self._addtree = addtree
        self._backend = backend
        self._strategy = strategy
        self._strategy.strategy_setup(self)
        self._prefix = prefix

        self.num_features = len(domains)
        self._xvars = [backend.add_real_var(f"{prefix}x{i}")
                for i in range(self.num_features)]
        self._wvars = [backend.add_real_var(f"{prefix}w{i}")
                for i in range(len(self._addtree))]
        self._rvars = {} # real additional variables
        self._bvars = {} # boolean additional variables
        self._fvar = backend.add_real_var(f"{prefix}f")

        self._status = Verifier.Result.UNKNOWN
        self._splits = None
        self._iteration_count = 0

        # initialize backend
        fexpr = SumExpr(self._addtree.base_score, *self._wvars)
        self._backend.add_constraint(fexpr == self._fvar)
        self._backend.add_constraint(InDomainConstraint(self, self._domains))

    def add_rvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._rvars
        rvar = self._backend.add_real_var(f"{self._prefix}r_{name}")
        self._rvars[name] = rvar

    def rvar(self, name):
        """ Get one of the additional decision variables. """
        return Rvar(self, name)

    def add_bvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._bvars
        bvar = self._backend.add_bool_var(f"{self._prefix}b_{name}")
        self._bvars[name] = bvar

    def bvar(self, name):
        """ Get one of the additional decision variables. """
        return Bvar(self, name)

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
        Add a user-defined constraint. Use add_rvar, rvar, bvar, xvar, and fvar
        to get access to the variables.
        """
        self._backend.add_constraint(constraint)

    def verify(self, constraint=True, timeout=3600 * 24 * 31):
        """
        Verify the model, i.e., try to find an assignment to the decision
        variables that
            (1) satisfies the constraints on
                - the input features (xvars)
                - the addtree output (fvar)
                - any additional decision variables (rvars and bvars)
            (2) satisfies the additive tree structure
            (3) satisfies the given constraint

        There are three possible outcomes:
            (1) Verifier.SAT, an assignment was found
            (2) Verifier.UNSAT, no assignment that satisfies the constraints
                possible
            (3) Verifier.UNKNOWN, the answer is unknown, e.g. because of
                timeout

        Subsequent calls to verify must reuse the state of the previous.
        """
        self._strategy.verify_setup()

        while True: # a do-while
            self._status = self._backend.check(constraint)

            if self._status != Verifier.Result.SAT: break
            if not self._strategy.verify_step(): break # add the next part of the problem encoding

            self._iteration_count += 1

        self._strategy.verify_teardown()
        return self._status

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
            "rs": { name => value } value map of additional real variables
            "bs": { name => value } value map of additional bool variables
            }
        """
        return self._backend.model(
                ("xs", self._xvars),
                ("ws", self._wvars),
                ("f", self._fvar),
                ("rs", self._rvars),
                ("bs", self._bvars))

    def exclude_model(self, model):
        """
        Mark the domain region inhabited by `model[xs]` as impossible.

        Usage for model sampling:
        ```
        while cond:
            status = verifier.verify()
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

    def _enc_tree(self, tree_index, tree, node = None):
        # - start at root
        # - if left/right reachable, recur -> strategy.get_reachability
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
            reachability = self._strategy.get_reachability(feat_id, split_value)
            l, r = False, False
            if reachability.covers(Verifier.Reachable.LEFT):
                l = self._enc_tree(tree_index, tree, left)
            if reachability.covers(Verifier.Reachable.RIGHT):
                r = self._enc_tree(tree_index, tree, right)
            return self._backend.encode_split(xvar, split_value, l, r)

    def _determine_tree_bounds(self, tree_index):
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
            reachability = self._strategy.get_reachability(feat_id, split_value)
            if reachability.covers(Verifier.Reachable.RIGHT):
                stack.append(tree.right(node))
            if reachability.covers(Verifier.Reachable.LEFT):
                stack.append(tree.left(node))

        return (lo, hi)




# -----------------------------------------------------------------------------


class SplitCheckStrategy(VerifierStrategy):
    """
    A strategy that ignores unreachable branches by check individual split
    conditions.
    """

    def strategy_setup(self, verifier):
        self._verifier = verifier

        self._m = len(self._verifier._addtree)
        self._reachability = {}
        self._remaining_trees = None
        self._bounds = [(-math.inf, math.inf)] * self._m

    def get_reachability(self, feat_id, split_value):
        p = (feat_id, split_value)
        if p in self._reachability:
            return self._reachability[p]
        return Verifier.Reachable.BOTH

    def verify_setup(self):
        if self._remaining_trees is not None:
            return

        self._remaining_trees = list(range(self._m))
        new_bounds = [self._verifier._determine_tree_bounds(i)
                for i in range(self._m)]

        for tree_index in range(self._m):
            old = self._bounds[tree_index]
            lo, hi = new_bounds[tree_index]

            if old == (lo, hi): continue

            # Add tree bound constraint to backend
            wvar = self._verifier.wvar(tree_index)
            self._verifier._backend.add_constraint((wvar >= lo) & (wvar <= hi))

        self._bounds = new_bounds

        # TODO sort by better heuristic
        bnd = lambda i: self._bounds[i]
        self._remaining_trees.sort(key=lambda i: max(abs(bnd(i)[0]), abs(bnd(i)[1])))

        self._test_addtree_reachability()

    def verify_step(self):
        if len(self._remaining_trees) == 0:
            return False

        tree_index = self._remaining_trees.pop()
        tree = self._verifier._addtree[tree_index]
        enc = self._verifier._enc_tree(tree_index, tree, tree.root())
        self._verifier._backend.add_constraint(enc)
        return True

    def verify_teardown(self):
        pass

    # -- private --

    def _test_addtree_reachability(self):
        """
        For each tree in the addtree, for each internal node in the tree,
        check which side of the split is reachable given the constraints, not
        considering the addtree.

        For solvers, implement `test_split_reachability`.
        """
        for tree_index in range(self._m):
            tree = self._verifier._addtree[tree_index]
            self._test_tree_reachability(tree)

    def _test_tree_reachability(self, tree):
        """ Test the reachability of the nodes in a single tree.  """
        stack = [(tree.root())]
        while len(stack) > 0:
            node = stack.pop()

            if tree.is_leaf(node): continue

            feat_id, split_value = tree.get_split(node)
            reachability = self.get_reachability(feat_id, split_value)
            xvar = self._verifier.xvar(feat_id)

            if reachability.covers(Verifier.Reachable.LEFT):
                check = self._verifier._backend.check(xvar < split_value)
                if not check.is_sat():
                    reachability ^= Verifier.Reachable.LEFT # disable left
                else: stack.append(tree.left(node))
            if reachability == Verifier.Reachable.BOTH:          # if left is unreachable, then no ...
                check = self._verifier._backend.check(xvar >= split_value) # ... need to test, right is reachable!
                if not check.is_sat():
                    reachability ^= Verifier.Reachable.RIGHT # disable right
                else: stack.append(tree.right(node))

            self._reachability[(feat_id, split_value)] = reachability





# -----------------------------------------------------------------------------

#class SplitCheckVerifier(Verifier):
#    """
#    A verifier that skips branches when the last split condition leading to
#    that branch is UNSAT given the user provided constraints.
#    """
#    def __init__(self, domains, addtree, backend, prefix=""):
#        super().__init__(domains, addtree, backend, prefix=prefix)
#
#        # (feat_id, split_value) => REACHABILITY FLAG
#        self._reachability = {}
#        self._remaining_trees = None
#        self._iteration_count = 0
#
#    def verify(self, constraint=True, timeout=3600 * 24 * 31):
#        # initialize the verifier if not done yet
#        if self._remaining_trees is None:
#            self._initialize()
#
#    def _verify_generator(self, constraint, timeout):
#        pass
#
#    def _initialize(self):
#        super()._initialize()
#
#        m = len(self._addtree)
#        self._remaining_trees = list(range(m))
#        self._bounds = [self._determine_tree_bounds(i) for i in range(m)]
#        self._add_tree_bounds_constraints()
#
#    def get_reachability(self, feat_id, split_value):
#        """ Check the reachability of the given split. """
#        p = (feat_id, split_value)
#        if p in self._reachability:
#            return self._reachability[p]
#        return Verifier.Reachable.BOTH
#
#    def get_reachability_dict(self):
#        """ Get the full unreachable dictionary for reuse in deeper verifiers. """
#        return self._reachability.copy()
#
#    def set_reachability_dict(self, reachability):
#        self._reachability = reachability
#
#    def _test_addtree_reachability(self):
#        """
#        For each tree in the addtree, for each internal node in the tree,
#        check which side of the split is reachable given the constraints, not
#        considering the addtree.
#
#        For solvers, implement `test_split_reachability`.
#        """
#        for tree_index in range(len(self._addtree)):
#            tree = self._addtree[tree_index]
#            self._test_tree_reachability(tree)
#
#    def _test_tree_reachability(self, tree):
#        """ Test the reachability of the nodes in a single tree.  """
#        stack = [(tree.root())]
#        while len(stack) > 0:
#            node = stack.pop()
#
#            if tree.is_leaf(node): continue
#
#            feat_id, split_value = tree.get_split(node)
#            reachability = self.get_reachability(feat_id, split_value)
#            xvar = self.xvar(feat_id)
#
#            if reachability.covers(Verifier.Reachable.LEFT):
#                check = self._backend.check(xvar < split_value)
#                if not check.is_sat():
#                    reachability ^= Verifier.Reachable.LEFT # disable left
#                else: stack.append(tree.left(node))
#            if reachability == Verifier.Reachable.BOTH:          # if left is unreachable, then no ...
#                check = self._backend.check(xvar >= split_value) # ... need to test, right is reachable!
#                if not check.is_sat():
#                    reachability ^= Verifier.Reachable.RIGHT # disable right
#                else: stack.append(tree.right(node))
#
#            self._reachability[(feat_id, split_value)] = reachability
#
#    #def _initialize(self):
#    #    super()._initialize()
#    #    self._test_addtree_reachability()
#    #    self._backend.simplify()
#
#class DefaultVerifier(Verifier):
#
#    def __init__(self, domains, addtree, backend):
#        super().__init__(domains, addtree, backend)
#        self._status = Verifier.Result.UNKNOWN
#
#    def verify(self, constraint=True, timeout=3600 * 24 * 31):
#        """
#        DefaultVerifier algorithm:
#         - define f as sum of ws          |
#         - add domain constraints         |  -> helper method in Verifier, e.g. initialize()
#         - add user defined constraints   |
#         - compute reachabilities         |
#         - add bounds for trees
#                -> get_reachability
#         - add full encodings of trees in order of best bounds -> get_reachability + backend
#                -> helper method enc_tree
#         - stop early when UNSAT
#         - stop with SAT if SAT with all trees fully encoded
#         - stop with UNKNOWN if backend times out
#        """
#
#        if reset or self._status != Verifier.Result.SAT:
#            self.reset()
#        self._status = self._backend.check(constraint)
#
#        while len(self._remaining_trees) > 0 \
#                and self._status == Verifier.Result.SAT:
#            self._iteration_count += 1
#
#            # TODO choose best tree according to heuristic
#            tree_index = self._remaining_trees.pop()
#            tree = self._addtree[tree_index]
#            enc = self._enc_tree(tree_index, tree, tree.root())
#            self._backend.add_constraint(enc)
#
#            self._status = self._backend.check(constraint)
#
#        return self._status
#
#    def reset(self):
#        m = len(self._addtree)
#        self._iteration_count = 0
#        self._remaining_trees = list(range(m))
#
#        self._initialize() # unreachable available now
#        self._bounds = [self._determine_tree_bounds(i) for i in range(m)]
#        self._add_bounds_constraints()
#
#        # TODO let user choose heuristic
#        # sort remaining trees by heuristic value
#        bnd = lambda i: self._bounds[i]
#        self._remaining_trees.sort(key=lambda i: max(abs(bnd(i)[0]), abs(bnd(i)[1])))
#
#        self._status = Verifier.Result.SAT
#
#
#class MultipleInstanceVerifier(Verifier):
#
#    def __init__(self, domains, addtree, backend,
#            num_instances=2,
#            subverifier=DefaultVerifier):
#        self._verifiers = [subverifier(domains, addtree, backend, prefix=str(i))
#            for i in range(num_instances)]
#
#    def __getitem__(self, verifier_index):
#        return self._verifiers[verifier_index]
#
#    def verify(self, constraint=True, timeout=3600*24*31):
#        pass
