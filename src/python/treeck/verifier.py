import math, timeit
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

# TODO remove
#class InDomainConstraint(VerifierAndExpr):
#    def __init__(self, verifier, domains):
#        cs = []
#        for feat_id, d in domains:
#            var = verifier.xvar(feat_id)
#            if math.isinf(d.lo) and math.isinf(d.hi):
#                continue # unconstrained feature -> everything possible
#            elif math.isinf(d.lo): cs.append(var <  d.hi)
#            elif math.isinf(d.hi): cs.append(var >= d.lo)
#            else:
#                cs.append(var >= d.lo)
#                cs.append(var <  d.hi)
#        super().__init__(*cs)

class NotInDomainConstraint(VerifierOrExpr):
    def __init__(self, verifier, domains):
        cs = []
        for feat_id, d in domains:
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


# TODO remove
#class VerifierStrategy:
#
#    def strategy_setup(self, verifier):
#        """
#        Setup circular dependency between Strategy and Verifier. Verifier is
#        responsible for calling this in its constructor.
#        """
#        pass
#
#    def get_reachability(self, tree, node):
#        """ Given the split, can we go left, right, or both?  """
#        return Verifier.Reachable.BOTH
#
#    def verify_setup(self):
#        """
#        Before starting the verification loop over the verification steps, this
#        method is called by the verifier.
#        """
#        pass
#
#    def verify_step(self):
#        """
#        This adds the next batch of constraints to the backend. Returns True if
#        there is new work to verify, or False if finished.
#        """
#        raise RuntimeError("abstract method")
#
#    def verify_teardown(self):
#        """
#        Called after the verification loop, regardless of outcome. Check
#        outcome in `verifier._status`.
#        """
#        pass



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

    def __init__(self, addtree, splittree_leaf, backend, suffix=""):
        self._addtree = addtree
        self._stree = splittree_leaf
        self._backend = backend
        self._suffix = suffix

        self._xvars = [backend.add_real_var(f"x{i}{suffix}")
                for i in range(addtree.num_features())]
        self._wvars = [backend.add_real_var(f"w{i}{suffix}")
                for i in range(len(self._addtree))]
        self._rvars = {} # real additional variables
        self._bvars = {} # boolean additional variables
        self._fvar = backend.add_real_var(f"f{suffix}")

        self._status = Verifier.Result.UNKNOWN
        self._splits = None

        self.check_time = -math.inf
        self.nchecks = 0

        # initialize backend
        fexpr = SumExpr(self._addtree.base_score, *self._wvars)
        self._backend.add_constraint(fexpr == self._fvar)

    def add_rvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._rvars
        rvar = self._backend.add_real_var(f"r_{name}{self._suffix}")
        self._rvars[name] = rvar

    def rvar(self, name):
        """ Get one of the additional decision variables. """
        return Rvar(self, name)

    def add_bvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._bvars
        bvar = self._backend.add_bool_var(f"b_{name}{self._suffix}")
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

    def set_timeout(self, timeout):
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

    def add_tree(self, tree_index):
        """ Add the full encoding of a tree to the verifier.  """
        tree = self._addtree[tree_index]
        enc = self._enc_tree(tree, tree.root())
        self.add_constraint(enc)
    
    def add_all_trees(self):
        """ Add all trees in the addtree. """
        for tree_index in range(len(self._addtree)):
            self.add_tree(tree_index)

    def model_family(self):
        # TODO implement, then remove exclude model and replace by add_constraint(..)
        # e.g. v.add_constraint(StrictModelExcludeConstraint(mf))
        #      v.add_constraint(RelaxedModelExcludeConstraint(mf))
        pass

    # TODO see above
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
        for feat_id, _, lo, hi in self._find_sample_intervals(model):
            d = RealDomain(lo, hi)
            if d.is_everything():
                raise RuntimeError("Unconstrained feature!")
            #print("{:.6g} <= {:.6g} < {:.6g}".format(lo, x, hi))
            domains.append((feat_id, d))
        self._backend.add_constraint(NotInDomainConstraint(self, domains))

    def _find_sample_intervals(self, model): # helper `exclude_assignment`
        for i, x in enumerate(model["xs"]):
            if x == None: continue
            if i not in self._splits: continue # feature not used in splits of trees
            split_values = self._splits[i]
            j = bisect(split_values, x)
            lo = -math.inf if j == 0 else split_values[j-1]
            hi = math.inf if j == len(split_values) else split_values[j]
            assert lo < hi
            assert x >= lo
            assert x < hi

            yield i, x, lo, hi

    def _enc_tree(self, tree, node):
        if tree.is_leaf(node):
            wvar = self._wvars[tree.index()]
            leaf_value = tree.get_leaf_value(node)
            return self._backend.encode_leaf(wvar, leaf_value)
        else:
            tree_index = tree.index()
            feat_id, split_value = tree.get_split(node)
            xvar = self._xvars[feat_id]
            left, right = tree.left(node), tree.right(node)
            l, r = False, False
            if self._stree.is_reachable(tree_index, left):
                l = self._enc_tree(tree, left)
            if self._stree.is_reachable(tree_index, right):
                r = self._enc_tree(tree, right)
            return self._backend.encode_split(xvar, split_value, l, r)

# TODO remove -> SplitTreeLeaf::get_tree_bounds
#    def _determine_tree_bounds(self, tree_index):
#        tree = self._addtree[tree_index]
#        lo =  math.inf
#        hi = -math.inf
#
#        stack = [tree.root()]
#        while len(stack) > 0:
#            node = stack.pop()
#
#            if tree.is_leaf(node):
#                leaf_value = tree.get_leaf_value(node)
#                lo = min(lo, leaf_value)
#                hi = max(hi, leaf_value)
#                continue
#
#            feat_id, split_value = tree.get_split(node)
#            reachability = self._strategy.get_reachability(tree, node)
#            if reachability.covers(Verifier.Reachable.RIGHT):
#                stack.append(tree.right(node))
#            if reachability.covers(Verifier.Reachable.LEFT):
#                stack.append(tree.left(node))
#
#        return (lo, hi)





# -----------------------------------------------------------------------------


# TODO remove
#class SplitCheckStrategy(VerifierStrategy):
#    """
#    A strategy that ignores unreachable branches by check individual split
#    conditions.
#    """
#
#    def strategy_setup(self, verifier):
#        self._verifier = verifier
#
#        self._m = len(self._verifier._addtree)
#        self._reachability = {}
#        self._remaining_trees = None
#        self._bounds = [(-math.inf, math.inf)] * self._m
#
#    def get_reachability(self, tree, node):
#        p = tree.get_split(node)
#        if p in self._reachability:
#            return self._reachability[p]
#        return Verifier.Reachable.BOTH
#
#    def set_reachability(self, reachability):
#        """
#        Reuse reachability from strategy of 'parent' model -> are unchanged, no
#        need to recalculate.
#        """
#        self._reachability = reachability
#
#    def verify_setup(self):
#        if self._remaining_trees is not None:
#            return
#
#        if len(self._reachability) == 0:      # no reachability provided...
#            self._test_addtree_reachability() # ... so compute! (see set_reachability)
#
#        self._remaining_trees = list(range(self._m))
#        new_bounds = [self._verifier._determine_tree_bounds(i)
#                for i in range(self._m)]
#
#        for tree_index in range(self._m):
#            old = self._bounds[tree_index]
#            lo, hi = new_bounds[tree_index]
#
#            if old == (lo, hi): continue
#
#            # Add tree bound constraint to backend
#            wvar = self._verifier.wvar(tree_index)
#            self._verifier.add_constraint((wvar >= lo) & (wvar <= hi))
#
#        self._bounds = new_bounds
#
#        # TODO use better sort heuristic
#        bnd = lambda i: self._bounds[i]
#        self._remaining_trees.sort(
#                key=lambda i: max(abs(bnd(i)[0]), abs(bnd(i)[1])),
#                reverse=True)
#
#    def verify_step(self):
#        if len(self._remaining_trees) == 0:
#            return False
#
#        tree_index = self._remaining_trees.pop()
#        tree = self._verifier._addtree[tree_index]
#        enc = self._verifier._enc_tree(tree, tree.root())
#        self._verifier.add_constraint(enc)
#        return True
#
#    def verify_teardown(self):
#        pass
#
#    # -- private --
#
#    def _test_addtree_reachability(self):
#        """
#        For each tree in the addtree, for each internal node in the tree,
#        check which side of the split is reachable given the constraints, not
#        considering the addtree.
#
#        For solvers, implement `test_split_reachability`.
#        """
#        for tree_index in range(self._m):
#            tree = self._verifier._addtree[tree_index]
#            self._test_tree_reachability(tree)
#
#    def _test_tree_reachability(self, tree):
#        stack = [(tree.root())]
#        while len(stack) > 0:
#            node = stack.pop()
#
#            if tree.is_leaf(node): continue
#
#            reachability = self.get_reachability(tree, node)
#            feat_id, split_value = tree.get_split(node)
#            xvar = self._verifier.xvar(feat_id)
#
#            if reachability.covers(Verifier.Reachable.LEFT):
#                check = self._verifier._check(xvar < split_value)
#                if check == Verifier.Result.UNSAT:
#                    reachability ^= Verifier.Reachable.LEFT # disable left
#                else: stack.append(tree.left(node))
#            if reachability == Verifier.Reachable.BOTH:            # if left is unreachable, then no ...
#                check = self._verifier._check(xvar >= split_value) # ... need to test, right is reachable
#                if check == Verifier.Result.UNSAT:
#                    reachability ^= Verifier.Reachable.RIGHT # disable right
#                else: stack.append(tree.right(node))
#
#            self._reachability[(feat_id, split_value)] = reachability



# TODO remove
#class PathCheckStrategy(VerifierStrategy):
#
#    def strategy_setup(self, verifier):
#        self._verifier = verifier
#        self._unreachable = set()
#        self._m = len(self._verifier._addtree)
#        self._remaining_trees = list(range(self._m))
#        self._bounds = [(-math.inf, math.inf)] * self._m
#
#    def get_reachability(self, tree, node):
#        l = (tree.index(), tree.left(node))
#        r = (tree.index(), tree.right(node))
#        reachability = Verifier.Reachable.NONE
#        if l not in self._unreachable:
#            reachability |= Verifier.Reachable.LEFT
#        if r not in self._unreachable:
#            reachability |= Verifier.Reachable.RIGHT
#        return reachability
#
#    def verify_setup(self):
#        pass
#
#    def verify_step(self):
#        if len(self._remaining_trees) == 0:
#            return False
#
#        v = self._verifier
#        for tree_index in self._remaining_trees:
#            bounds_changed = self._test_tree_reachability(tree_index)
#            if bounds_changed:
#                wvar = v.wvar(tree_index)
#                lo, hi = self._bounds[tree_index]
#                v.add_constraint((wvar >= lo) & (wvar <= hi))
#
#        # TODO use better sort heuristic
#        bnd = lambda i: self._bounds[i]
#        self._remaining_trees.sort(
#                key=lambda i: max(abs(bnd(i)[0]), abs(bnd(i)[1])),
#                reverse=True)
#
#        # encode the best tree
#        tree_index = self._remaining_trees.pop()
#        tree = self._verifier._addtree[tree_index]
#        enc = self._verifier._enc_tree(tree, tree.root())
#        self._verifier.add_constraint(enc)
#        return True
#
#    def verify_teardown(self):
#        pass
#
#    # -- private --
#
#    def _test_tree_reachability(self, tree_index):
#        v = self._verifier
#        tree = v._addtree[tree_index]
#        stack = [(tree.root(), True)]
#
#        lo =  math.inf
#        hi = -math.inf
#
#        while len(stack) > 0:
#            node, path_constraint = stack.pop()
#
#            if tree.is_leaf(node):
#                leaf_value = tree.get_leaf_value(node)
#                lo = min(lo, leaf_value)
#                hi = max(hi, leaf_value)
#                continue
#
#            reachability = self.get_reachability(tree, node)
#            feat_id, split_value = tree.get_split(node)
#            xvar = v.xvar(feat_id)
#
#            if reachability.covers(Verifier.Reachable.LEFT):
#                left = tree.left(node)
#                c = (xvar < split_value) & path_constraint
#                check = v._check(c)
#                if check == Verifier.Result.SAT:
#                    stack.append((left, c))
#                else:
#                    self._unreachable.add((tree.index(), left))
#
#            if reachability.covers(Verifier.Reachable.RIGHT):
#                right = tree.right(node)
#                c = (xvar >= split_value) & path_constraint
#                check = v._check(c)
#                if check == Verifier.Result.SAT:
#                    stack.append((right, c))
#                else:
#                    self._unreachable.add((tree.index(), right))
#
#        changed = self._bounds[tree_index] != (lo, hi)
#        self._bounds[tree_index] = (lo, hi)
#
#        if changed:
#            print("{:4}: DEBUG new bounds tree{}: [{:.4g}, {:.4g}]".format(
#                v._iteration_count, tree.index(), lo, hi))
#        return changed



# -----------------------------------------------------------------------------


#class MultiInstanceVerifier:
#
#    def __init__(self, domains, addtree, backend,
#            num_instances=2,
#            strategy_factory=SplitCheckStrategy):
#        self._iteration_count = 0
#        self._status = Verifier.Result.UNKNOWN
#        self._backend = backend
#        self._verifiers = [Verifier(domains, addtree, backend,
#                                    strategy_factory(),
#                                    suffix=f"_{i}")
#            for i in range(num_instances)]
#
#    def __getitem__(self, verifier_index):
#        """
#        Use this to get the variables of the individual instance models to
#        construct constraints.
#        """
#        return self._verifiers[verifier_index]
#
#    def set_timeout(self, timeout):
#        self._backend.set_timeout(timeout)
#
#    def add_rvar(self, name): # just use the first verifier for these
#        self._verifiers[0].add_rvar(name)
#
#    def rvar(self, name):
#        return self._verifiers[0].rvar(name)
#
#    def add_bvar(self, name):
#        self._verifiers[0].add_bvar(name)
#
#    def bvar(self, name):
#        return self._verifiers[0].bvar(name)
#
#    def add_constraint(self, constraint):
#        self._backend.add_constraint(constraint)
#
#    def verify(self, constraint=True):
#        """
#        Exactly the same as other verifier, but add one tree from each instance
#        at a time.
#        """
#        for v in self._verifiers:
#            v._strategy.verify_setup()
#
#        while True: # a do-while
#            self._status = self._backend.check(constraint)
#
#            if self._status != Verifier.Result.SAT: break
#
#            ndone = 0
#            for v in self._verifiers:
#                if not v._strategy.verify_step():
#                    ndone += 1
#            assert ndone == 0 or ndone == len(self._verifiers)
#            if ndone > 0: break
#
#            self._backend.simplify()
#            self._iteration_count += 1
#
#        for v in self._verifiers:
#            v._strategy.verify_teardown()
#        return self._status
#
#    def model(self):
#        return [v.model() for v in self._verifiers]
#
#    def exclude_model(self, model):
#        for (m, v) in zip(model, self._verifiers):
#            v.exclude_model(m)

