
class VerifierConstraint:
    def construct(self, verifier):
        if isinstance(verifier, Z3Verifier):
            self.construct_z3(verifier)
        else:
            raise RuntimeError(f"unknown verifier type {type(verifier)}")

    def construct_z3(self, z3verifier):
        raise RuntimeError(f"construct_z3 not implemented for {type(self)}")



class Verifier:
    # TODO remove consts.py
    SAT = 0x11
    UNSAT = 0x22
    UNKNOWN = 0x33

    def __init__(self, constraints, domains, addtree):
        """
        Initialize a Verifier.
         - constraints is a list of `VerifierConstraint` objects.
         - domains is a list of `RealDomain` objects, one for each feature.
         - addtree is the model to verify properties of.
        """
        self._constraints = constraints
        self._domains = domains
        self._excluded_assignments = []

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
        # - use a ExcludeDomainConstraint <: VerifierConstraint constraint and
        #   insert into _excluded_assignments
        pass


