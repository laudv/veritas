import math, timeit
from bisect import bisect

from enum import Enum
from . import RealDomain, BoolDomain, AddTreeFeatureTypes
from . import DomTree, DomTreeLeaf

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

class Xvar(VerifierVar, VerifierRealExpr, VerifierBoolExpr): # can be both real/bool
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

def in_domain_constraint(verifier, domains, instance):
    cs = []
    for feat_id, dom in domains.items():
        if dom.is_everything():
            raise RuntimeError("Unconstrained feature -> should not be in dict")
        var = verifier.xvar(feat_id, instance=instance)
        if isinstance(dom, RealDomain):
            if   math.isinf(dom.lo): cs.append(var <  dom.hi)
            elif math.isinf(dom.hi): cs.append(var >= dom.lo)
            else:
                cs.append((var >= dom.lo) & (var < dom.hi))
        elif isinstance(dom, BoolDomain):
            if   dom.is_true():  cs.append(var)
            elif dom.is_false(): cs.append(VerifierNotExpr(var))
        else: raise RuntimeError(f"unknown domain type {type(dom)}")
    return VerifierAndExpr(*cs)

def not_in_domain_constraint(verifier, domains, instance, strict=True):
    cs = []
    for feat_id, dom in domains.items():
        if dom.is_everything():
            raise RuntimeError("Unconstrained feature -> nothing possible?")
        var = verifier.xvar(feat_id, instance=instance)
        if isinstance(dom, RealDomain):
            if   math.isinf(dom.lo): cs.append(var >= dom.hi)
            elif math.isinf(dom.hi): cs.append(var <  dom.lo)
            else:
                cs.append((var < dom.lo) | (var >= dom.hi))
        elif isinstance(dom, BoolDomain):
            if   dom.is_true():  cs.append(VerifierNotExpr(var))
            elif dom.is_false(): cs.append(var)
        else: raise RuntimeError(f"unknown domain type {type(dom)}")

    if strict:
        return VerifierOrExpr(*cs)
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

    def encode_split(self, feat_var, split, left, right):
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

    def __init__(self, domtree_leaf, backend):
        assert isinstance(backend, VerifierBackend)
        assert isinstance(domtree_leaf, DomTreeLeaf)

        self._backend = backend
        self._lk = domtree_leaf
        self._instances = [AddTreeInstance(self, i)
                for i in range(self._lk.num_instances())]

        self._rvars = {} # real additional variables
        self._bvars = {} # boolean additional variables

        self._status = Verifier.Result.UNKNOWN

        self.check_time = -math.inf
        self.nchecks = 0

    def instance(self, instance=0):
        return self._instances[instance]

    def xvar(self, feat_id, instance=0):
        return self.instance(instance).xvar(feat_id)

    def wvar(self, tree_index, instance=0):
        return self.instance(instance).wvar(tree_index)

    def fvar(self, instance=0):
        return self.instance(instance).fvar()

    def add_tree(self, tree_index, instance=0):
        self.instance(instance).add_tree(tree_index)

    def add_all_trees(self, instance=None):
        if instance is None:
            for inst in self._instances:
                inst.add_all_trees()
        else:
            self.instance(instance).add_all_trees()

    def add_rvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._rvars
        rvar = self._backend.add_real_var(f"r_{name}")
        self._rvars[name] = rvar

    def rvar(self, name):
        """ Get one of the additional decision variables. """
        return Rvar(self, name)

    def add_bvar(self, name):
        """ Add an additional decision variable to the problem. """
        assert name not in self._bvars
        bvar = self._backend.add_bool_var(f"b_{name}")
        self._bvars[name] = bvar

    def bvar(self, name):
        """ Get one of the additional decision variables. """
        return Bvar(self, name)

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
        if len(self._instances) == 1:
            args = [("xs", self._instances[0]._xvars),
                    ("ws", self._instances[0]._wvars),
                    ("f",  self._instances[0]._fvar )]
        else:
            args = [(i, [("xs", inst._xvars),
                         ("ws", inst._wvars),
                         ("f",  inst._fvar )])
                    for i, inst in enumerate(self._instances)]
        args.append(("rs", self._rvars))
        args.append(("bs", self._bvars))
        return self._backend.model(*args)

    def model_family(self, model):
        """
        Get ranges on the xvar values within which the model does not
        change its predicted value.
        """
        if len(self._instances) == 1:
            return self._instances[0]._xs_family(model["xs"])
        return [inst._xs_family(model[i]["xs"])
                for i, inst in enumerate(self._instances)]




# -----------------------------------------------------------------------------



class AddTreeInstance:

    def __init__(self, verifier, instance_index):
        self._v = verifier
        self._instance_index = instance_index
        self._addtree = self._v._lk.addtree(instance_index)
        self._feat_types = AddTreeFeatureTypes(self._addtree)

        suffix = f"_{instance_index}"

        self._xvars = {fid: self._v._backend.add_real_var(f"x{fid}{suffix}")
                if typ == "lt"
                else self._v._backend.add_bool_var(f"xb{fid}{suffix}")
                for fid, typ in self._feat_types}
        self._wvars = [self._v._backend.add_real_var(f"w{i}{suffix}")
                for i in range(len(self._addtree))]
        self._fvar = self._v._backend.add_real_var(f"f{suffix}")

        # FVAR = sum{WVARS}
        fexpr = SumExpr(self._addtree.base_score, *self._wvars)
        self._v.add_constraint(fexpr == self.fvar())

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

    def add_tree(self, tree_index):
        """ Add the full encoding of a tree to the backend.  """
        tree = self._addtree[tree_index]
        enc = self._enc_tree(tree, tree.root())
        self._v._backend.add_constraint(enc)

    def add_all_trees(self):
        """ Add all trees in the addtree. """
        for tree_index in range(len(self._addtree)):
            self.add_tree(tree_index)

    def feat_ids(self):
        """ Loop over all feature IDs in the associated addtree. """
        yield from self._feat_types.feat_ids()

    def mark_unreachable_paths(self, tree_index, only_feat_id = -1):
        """
        Check the reachability of the paths in the trees of this instance
        against the constraints in the Verifier.
        """
        i = self._instance_index
        tree = self._addtree[tree_index]
        stack = [(tree.root(), True)]

        while len(stack) > 0:
            node, path_constraints = stack.pop()

            l, r = tree.left(node), tree.right(node)
            split = tree.get_split(node) # (split_type, feat_id...)
            feat_id = split[1]
            xvar = self.xvar(feat_id)

            if only_feat_id != -1 and feat_id != only_feat_id:
                continue # only test paths splitting on this feat_id

            if split[0] == "lt":
                split_value = split[2]
                constraint_l = (xvar < split_value)
                constraint_r = (xvar >= split_value)
            elif split[0] == "bool":
                constraint_l = VerifierNotExpr(xvar) # false left, true right
                constraint_r = xvar
            else: raise RuntimeError(f"unknown split type {split[0]}")

            if self._v._lk.is_reachable(i, tree_index, l):
                path_constraints_l = constraint_l & path_constraints;
                if self._v.check(path_constraints_l).is_sat():
                    if tree.is_internal(l):
                        stack.append((l, path_constraints_l))
                else:
                    #print(f"unreachable  left: {i} {tree_index} {l}, {only_feat_id}")
                    self._v._lk.mark_unreachable(i, tree_index, l)

            if self._v._lk.is_reachable(i, tree_index, r):
                path_constraints_r = constraint_r & path_constraints;
                if self._v.check(path_constraints_r).is_sat():
                    if tree.is_internal(r):
                        stack.append((r, path_constraints_r))
                else:
                    #print(f"unreachable right: {i} {tree_index} {r}, {only_feat_id}")
                    self._v._lk.mark_unreachable(i, tree_index, r)

    def _enc_tree(self, tree, node):
        if tree.is_leaf(node):
            wvar = self._wvars[tree.index()]
            leaf_value = tree.get_leaf_value(node)
            self.leaf_count += 1
            return self._v._backend.encode_leaf(wvar, leaf_value)
        else:
            tree_index = tree.index()
            split = tree.get_split(node)
            xvar = self._xvars[split[1]]
            left, right = tree.left(node), tree.right(node)
            l, r = False, False
            if self._v._lk.is_reachable(self._instance_index, tree_index, left):
                l = self._enc_tree(tree, left)
            if self._v._lk.is_reachable(self._instance_index, tree_index, right):
                r = self._enc_tree(tree, right)
            return self._v._backend.encode_split(xvar, split, l, r)

    def _xs_family(self, xs):
        if self._splits is None:
            self._splits = self._addtree.get_splits()

        domains = {}
        for feat_id, x, dom in self._find_sample_intervals(xs):
            if dom.is_everything():
                raise RuntimeError("Unconstrained feature!")
            #print("[feat_id={:<3}] {} in {}".format(feat_id, x, dom))
            domains[feat_id] = dom
        return domains

    def _find_sample_intervals(self, xs):
        assert isinstance(xs, dict)
        for feat_id, x in xs.items():
            if x == None: continue
            ftype = self._feat_types[feat_id]
            if ftype == "lt":
                if feat_id not in self._splits: continue # feature not used in splits of trees
                split_values = self._splits[feat_id]
                j = bisect(split_values, x)
                lo = -math.inf if j == 0 else split_values[j-1]
                hi = math.inf if j == len(split_values) else split_values[j]
                assert lo < hi
                assert x >= lo
                assert x < hi
                dom = RealDomain(lo, hi)
            elif ftype == "bool":
                dom = BoolDomain(x)
            else:
                raise RuntimeError("unknown ftype")
            yield feat_id, x, dom

    def _xs_wide_family(self, xs):
        domains = {}
        for tree_index, tree in enumerate(self._addtree):
            node = tree.predict_leaf(xs)
            print(tree_index, leaf_id)

            # TODO complete
            # scan all paths of tree to compute much less restricted family
            # idea: as long as the features vary within their domains, the
            #    prediction is going to remain unchanged
            # _xs_family based on all splits is too strict
            while not tree.is_root(node):
                node = tree.parent(node)
                split = tree.get_split(node)
                feat_id = split[1]

                if split[0] == "lt":
                    dom = domains.get(feat_id, RealDomain())
                elif split[0] == "bool":
                    dom = domains.get(feat_id, BoolDomain())
                else: raise RuntimeError("unknown split")

                if tree.is_root(node):
                    break





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
