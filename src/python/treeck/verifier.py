import math
import bisect

from collections import defaultdict


class VerifierConstraint:
    def construct(self, verifier):
        if isinstance(verifier, Z3Verifier):
            self.construct_z3(verifier)
        else:
            raise RuntimeError(f"unknown verifier type {type(verifier)}")

    def construct_z3(self, z3verifier):
        raise RuntimeError(f"construct_z3 not implemented for {type(self)}")

class ExcludeAssignmentConstraint(VerifierConstraint):
    def __init__(self, domains):
        self.domains = domains

    #def construct_z3(self, verifier): # --> move to Z3 implementation!
    #    #elif math.isinf(lo): d.hi = hi
    #    #elif math.isinf(hi): c = (var < lo)
    #    #else: c = z3.Or((var < lo), (var >= hi))
    #    constraints = []
    #    for d in self._domains:
    #        if math.isinf(d.lo) and math.isinf(d.hi):
    #            raise RuntimeError("Unconstrained feature")
    #        elif math.isinf(d.lo): c = (var >= hi)
    #        elif math.isinf(d.hi): c = (var < lo)
    #        else: c = z3.Or((var < lo), (var >= hi))





class Verifier:
    # TODO remove consts.py
    SAT = 0x11
    UNSAT = 0x22
    UNKNOWN = 0x33

    BOTH_UNREACHABLE = 0x0
    LEFT_REACHABLE = 0x1
    RIGHT_REACHABLE = 0x2
    BOTH_REACHABLE = LEFT_REACHABLE | RIGHT_REACHABLE

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
        self._unreachable = defaultdict(lambda: BOTH_REACHABLE)

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

    def verify(self, timeout=3600 * 24 * 31)
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
            test_tree_reachability(tree)

    def test_tree_reachability(self, tree):
        """ Test the reachability of the nodes in a single tree.  """
        stack = [(tree.root())]
        while len(stack) > 0:
            node = stack.pop()

            if tree.is_leaf(node): continue

            feat_id, split_value = tree.get_split(node)
            reachability = self.test_split_reachability(feat_id, split_value)

            assert reachability != BOTH_UNREACHABLE # this would be a bug

            if reachability & LEFT_REACHABLE:
                stack.append(tree.left(node))
            if reachability & RIGHT_REACHABLE:
                stack.append(tree.right(node))

    def test_split_reachability(self, feat_id, split_value):
        """
        Test whether the constraints -- not including the model itself --
        allows the left/right branch of this split.

        Returns a reachability flag:
            - LEFT_REACHABLE
            - RIGHT_REACHABLE
            - BOTH_REACHABLE = LEFT_REACHABLE | RIGHT_REACHABLE
        """
        raise RuntimeError(f"test_split_reachability not implemented for {type(self)}")
