import math
import bisect

from enum import Enum
from collections import defaultdict


class ConstraintVar:
    def get(self, verifier):
        raise RuntimeError("not implemented")

    def __lt__(self, other):
        if not isinstance(other, ConstraintVar):
            other = Cvar(other)
        return LtConstraint(self, other)

class Xvar(ConstraintVar):
    def __init__(self, feat_id):
        self._feat_id = feat_id
    def get(self, verifier):
        return verifier.xvar(self._feat_id)

class Dvar(ConstraintVar):
    def __init__(self, name):
        self._name = name
    def get(self, verifier):
        return verifier.dvar(self._name)

class Fvar(ConstraintVar):
    def get(self, verifier):
        return verifier.fvar()

class Cvar(ConstraintVar):
    def __init__(self, constant):
        self._constant = constant

    def get(self, verifier):
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
        raise RuntimeError("not implemented")

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
        raise RuntimeError("not implemented")

    def add_var(self, name):
        """ Add a new variable to the session. """
        raise RuntimeError("not implemented")

    def add_constraint(self, constraint, verifier=None):
        """
        Add a constraint to the current session. Constraint can be a
        VerifierConstraint or a Backend specific constraint. A valid verifier
        is needed to evaluate the ConstraintVars in the case of a
        VerifierConstraint
        """
        raise RuntimeError("not implemented")

    def encode_leaf(self, tree_var, leaf_value):
        """ Encode the leaf node """
        raise RuntimeError("not implemented")

    def encode_split(self, feat_var, split_value, left, right):
        """
        Encode the given split using left and right as the encodings of the
        subtrees.
        """
        raise RuntimeError("not implemented")

    def check(self, *constraints):
        """ Satisfiability check, optionally with additional constraints. """
        raise RuntimeError("not implemented")

    def model(self, *name_vars_pairs):
        """
        Get assignment to the given variables. The format of name_vars_pairs is:
            `(name1, [var1, var2, ...]), (name2, var), ...`
        Returns a dictionary:
            { name1: [var1_value, var2_value, ...], name2: var_value, ... }
        """
        raise RuntimeError("not implemented")



class VerifierStrategy:
    def verify(self, timeout):
        raise RuntimeError("not implemented")


class Verifier:
    class Result(Enum):
        SAT = 1
        UNSAT = 0
        UNKNOWN = -1

    class Reachable(Enum):
        NONE = 0x0
        LEFT = 0x1
        RIGHT = 0x2
        BOTH = 0x1 | 0x2

        def covers(self, other):
            return self.value & other.value > 0

    def __init__(self, constraints, domains, addtree):
        """
        Initialize a Verifier.
         - constraints is a list of `VerifierConstraint` objects.
         - domains is a list of `RealDomain` objects, one for each feature.
         - addtree is the model to verify properties of.
        """
        self._constraints = constraints
        self._domains = domains
        self._addtree = addtree
        self._excluded_assignments = []
        self._splits = None

        # (feat_id, split_value) => REACHABILITY FLAG
        self._unreachable = defaultdict(lambda: Verifier.Reachable.BOTH)

    def add_dvar(self, name):
        """ Add an additional decision variable to the problem. """
        raise RuntimeError(f"add_dvar not implemented for {type(self)}")

    def dvar(self, name):
        """ Get one of the additional decision variables. """
        raise RuntimeError(f"dvar not implemented for {type(self)}")

    def xvar(self, feat_id):
        """ Get the decision variable associated with feature `feat_id`. """
        raise RuntimeError(f"xvar not implemented for {type(self)}")

    def fvar(self):
        """ Get the decision variable associated with the output of the model. """
        raise RuntimeError(f"fvar not implemented for {type(self)}")

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
        raise RuntimeError(f"verify not implemented for {type(self)}")

    def get_assignment(self):
        """
        If a call to `verify` was successful, i.e., the output was
        `Verifier.SAT`, then this method returns a `dict` with the following structure:

        dict{
            "xs": [ list of xvar values ],
            "ws": [ list of tree leaf weights ],
            "f": sum of tree leaf weights == addtree.base_score + sum{ws}
            "ds": { name => value } value map of additional variables
            }
        """
        raise RuntimeError(f"get_assignment not implemented for {type(self)}")

    def exclude_assignment(self, assignment):
        """
        Mark the domain region inhabited by `assignment[xs]` as impossible.

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
        # TODO implement here! -> self._excluded_assignments
        # TODO move searchspace.cpp:extract_splits into addtree and expose to python
        # - use these splits here to define "domain region inhabited by ..."
        # - use a ExcludeAssignmentConstraint <: VerifierConstraint constraint and
        #   insert into _excluded_assignments
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
        return self._unreachable((feat_id, split_value))

    def get_unreachable_dict(self):
        """ Get the full unreachable dictionary for reuse in deeper verifiers. """
        return self._unreachable.copy()

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
            reachability = self.test_split_reachability(feat_id, split_value)

            assert reachability != Verifier.Reachable.NONE # this would be a bug

            if reachability.covers(Verifier.Reachable.LEFT):
                stack.append(tree.left(node))
            if reachability.covers(Verifier.Reachable.RIGHT):
                stack.append(tree.right(node))

    def test_split_reachability(self, feat_id, split_value):
        """
        Test whether the constraints -- not including the model itself --
        allows the left/right branch of this split.

        Returns a reachability flag:
            - Reachable.LEFT
            - Reachable.RIGHT
            - Reachable.BOTH = Reachable.LEFT | Reachable.RIGHT
        """
        raise RuntimeError(f"test_split_reachability not implemented for {type(self)}")
