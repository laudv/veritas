import math
import bisect

from enum import Enum


class ConstraintVar:
    def __init__(self, verifier):
        self._verifier = verifier

    def get(self):
        raise RuntimeError("abstract method")

class Xvar(ConstraintVar):
    def __init__(self, verifier, feat_id):
        super().__init__(verifier)
        self._feat_id = feat_id

    def get(self):
        return self._verifier._xvars[self._feat_id]

class Dvar(ConstraintVar):
    def __init__(self, verifier, name):
        super().__init__(verifier)
        self._name = name

    def get(self):
        return self._verifier._dvars[self._name]

class Fvar(ConstraintVar):
    def __init__(verifier):
        super().__init__(verifier)

    def get(self):
        return self._verifier._fvar

class Cvar(ConstraintVar):
    def __init__(self, constant):
        super().__init__(None)
        self._constant = constant

    def get(self):
        return self._constant


class VerifierConstraint:
    pass

FUNDAMENTAL_CONSTRAINTS = [
    ("LtConstraint", "__lt__"),
    ("GtConstraint", "__gt__"),
    ("LeConstraint", "__le__"),
    ("GeConstraint", "__ge__"),
    ("EqConstraint", "__eq__"),
    ("NeConstraint", "__ne__")]

for (clazz, method) in FUNDAMENTAL_CONSTRAINTS:
    exec(f"""
class {clazz}(VerifierConstraint):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
""")
    locs = {"f": None}
    exec(f"""
def f(self, other):
    if not isinstance(other, ConstraintVar):
        other = Cvar(other)
    return {clazz}(self, other)
""", globals(), locs)
    setattr(ConstraintVar, method, locs["f"])

class CompoundVerifierConstraint(VerifierConstraint):
    def compounds(self):
        """
        A generator of `VerifierConstraint` compounds, each of them should hold
        (i.e. logical AND).
        """
        raise RuntimeError("abstract method")

class ExcludeAssignmentConstraint(CompoundVerifierConstraint):
    def __init__(self, domains):
        self.domains = domains

    def compounds(self):
        for feat_id, d in enumerate(self.domains):
            var = Xvar(feat_id)
            if math.isinf(d.lo) and math.isinf(d.hi):
                raise RuntimeError("Unconstrained feature")
            elif math.isinf(d.lo): yield (var >= d.hi)
            elif math.isinf(d.hi): yield (var < d.lo)
            else:
                yield (var < d.lo)
                yield (var >= d.lo)


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
        VerifierConstraint or a Backend specific constraint. A valid verifier
        is needed to evaluate the ConstraintVars in the case of a
        VerifierConstraint
        """
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
            if self == Result.UNKNOWN:
                raise RuntimeError("unhandled Result.UNKNOWN")
            return self == Result.SAT

    class Reachable(Enum):
        NONE = 0x0
        LEFT = 0x1
        RIGHT = 0x2
        BOTH = 0x1 | 0x2

        def covers(self, other):
            return self.value & other.value > 0

        def __or__(self, other):
            return Reachable(self.value | other.value)

        def __xor__(self, other):
            return Reachable(self.value ^ other.value)

    def __init__(self, constraints, domains, addtree, backend):
        """
        Initialize a Verifier.
         - constraints is a list of `VerifierConstraint` objects.
         - domains is a list of `RealDomain` objects, one for each feature.
         - addtree is the model to verify properties of.
        """
        self._constraints = constraints
        self._domains = domains
        self._addtree = addtree
        self._backend = backend

        self._num_features = len(domains)
        self._xvars = [backend.add_var(f"x{i}") for i in range(self._num_features)]
        self._xvars = [backend.add_var(f"w{i}") for i in range(len(self._addtree))]
        self._dvars = {}
        self._fvar = backend.add_var("f")

        self._excluded_assignments = []
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

    def fvar(self):
        """ Get the decision variable associated with the output of the model. """
        return Fvar(self)

    def verify(self, timeout=3600 * 24 * 31):
        """
        Verify the model, i.e., try to find an assignment to the decision variables that
            (1) satisfies the constraints on
                - the input features (xvars)
                - the addtree output (fvar)
                - any additional decision variables (dvars)
            (2) satisfies the additive tree structure

        There are three possible outcomes:
            (1) Verifier.SAT, an assignment was found
            (2) Verifier.UNSAT, no assignment that satisfies the constraints possible
            (3) Verifier.UNKNOWN, the answer is unknown, e.g. because of timeout
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

        Usage for assignment sampling:
        ```
        while cond:
            status = verifier.verify()
            if status != Verifier.SAT: break
            assignment = verifier.get_assignment()
            # DO SOMETHING WITH assignment
            verifier.exclude_assignment(assignment)
        ```
        """
        if self._splits is None:
            self._splits = self._addtree.get_splits()

        domains = []
        for i, _, lo, hi in self._find_sample_intervals(assignment):
            d = RealDomain(lo, hi)
            if d.is_everything():
                raise RuntimeError("Unconstrained feature!")
            #print("{:.6g} <= {:.6g} < {:.6g}".format(lo, x, hi))
            domains.append(d)
        self._excluded_assignments.append(ExcludeAssignmentConstraint(domains))

    def _find_sample_intervals(self, assignment): # helper `exclude_assignment`
        for i, x in enumerate(assignment["xs"]):
            if x == None: continue
            split_values = self._splits[i]
            j = bisect(split_values, x)
            lo = -math.inf if j == 0 else split_values[j-1]
            hi = math.inf if j == len(split_values) else split_values[j]
            assert lo < hi
            assert x >= lo
            assert x < hi

            yield i, x, lo, hi

    def clear_excluded_assignments(self):
        """ Remove all previously excluded assignments added using `exclude_assignment`. """
        self._excluded_assignments.clear()

    def get_reachability(self, feat_id, split_value):
        """ Check the reachability of the given split. """
        p = (feat_id, split_value)
        if p in self._reachability:
            return self.reachability(p)
        return Verifier.Reachable.BOTH

    def get_reachability_dict(self):
        """ Get the full unreachable dictionary for reuse in deeper verifiers. """
        return self._reachability.copy()

    def test_addtree_reachability(self):
        """
        For each tree in the addtree, for each internal node in the tree,
        check which side of the split is reachable given the constraints, not
        considering the addtree.

        For solvers, implement `test_split_reachability`.
        """
        for tree_index in range(len(self._addtree)):
            tree = self._addtree[tree_index]
            self.test_tree_reachability(tree)

    def test_tree_reachability(self, tree):
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
