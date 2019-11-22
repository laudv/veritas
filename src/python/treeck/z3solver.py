import math
import z3

from .consts import LESS_THAN, GREATER_THAN

class Z3Solver:

    def __init__(self, num_features, domains, pytrees):
        self._num_features = num_features
        self._domains = domains
        self._trees = pytrees

        self._xvars = [z3.Real("x{}".format(i)) for i in range(self._num_features)]
        self._wvars = [z3.Real("w{}".format(i)) for i in range(len(self._trees))]

    def xvar(self, i):
        return self._xvars[i]

    def trees(self):
        yield from self._trees

    def verify(self, constraints=[], threshold=0.0, op=LESS_THAN):
        self._remaining_trees = list(range(len(self._trees)))
        self._lobounds = [ math.inf] * len(self._trees)
        self._upbounds = [-math.inf] * len(self._trees)
        self._iteration_count = 0

        self._solver = z3.Solver()
        self._solver.add(*constraints)
        self._solver.add(*self._domain_constraints())
        self._solver.add(self._output_constraint(threshold, op))

        status = z3.sat

        while len(self._remaining_trees) > 0 and status == z3.sat:
            self._iteration_count += 1

            # Compute bounds and reachable branches for all remaining trees
            for tree in self._remaining_trees:
                bounds_changed = self.test_tree_reachability(tree)
                if bounds_changed:
                    self._solver.add(self._encode_tree_bound(tree))

            # Add the tree with the best bound
            best_tree = self._pop_best_tree(op)
            enc = self._encode_tree(best_tree, self._trees.root(best_tree))
            self._solver.add(enc)

            status = self._solver.check()
            print()
            print("{:4}: DEBUG added tree{} with bounds [{:.4g}, {:.4g}] -> {}".format(
                        self._iteration_count, best_tree,
                        self._lobounds[best_tree],
                        self._upbounds[best_tree], status))

        print(self._solver)
        print("status:", status)
        return status

    def test_tree_reachability(self, tree):
        trees = self._trees
        stack = [(trees.root(tree), True)]

        lo =  math.inf
        hi = -math.inf

        while len(stack) > 0:
            node, path_constraint = stack.pop()

            if not self._trees.is_reachable(tree, node): continue

            if trees.is_leaf(tree, node):
                leaf_value = trees.leaf_value(tree, node)
                lo = min(lo, leaf_value)
                hi = max(hi, leaf_value)
                continue

            split = self._get_split_constraint(tree, node)
            lpath_constraint = z3.And(path_constraint, split)
            rpath_constraint = z3.And(path_constraint, z3.Not(split))

            l, r = trees.left(tree, node), trees.right(tree, node)
            lreachable = self._trees.is_reachable(tree, l)
            rreachable = self._trees.is_reachable(tree, r)

            # RIGHT
            if rreachable:
                if self._solver_check(rpath_constraint):
                    stack.append((r, rpath_constraint))
                else:
                    print("{:4}: DEBUG unreachable right tree{}: {}->{}".format(
                        self._iteration_count, tree, node, r))
                    self._trees.set_unreachable(tree, r)
                    assert not self._trees.is_reachable(tree, r)

            # LEFT
            if lreachable:
                if self._solver_check(lpath_constraint):
                    stack.append((l, lpath_constraint))
                else:
                    print("{:4}: DEBUG unreachable left tree{}: {}->{}".format(
                        self._iteration_count, tree, node, l))
                    self._trees.set_unreachable(tree, l)
                    assert not self._trees.is_reachable(tree, l)

        changed = self._lobounds[tree] != lo or self._upbounds[tree] != hi

        self._lobounds[tree] = lo
        self._upbounds[tree] = hi

        if changed:
            print("{:4}: DEBUG new bounds tree{}: [{:.4g}, {:.4g}]".format(
                self._iteration_count, tree,
                self._lobounds[tree], self._upbounds[tree]))
        return changed

    def _domain_constraints(self):
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

    def _output_constraint(self, threshold, op):
        sum_of_weights = self._wvars[0]
        for i in range(1, len(self._wvars)):
            sum_of_weights = sum_of_weights + self._wvars[i]
        if   op == LESS_THAN:    return (sum_of_weights < threshold)
        elif op == GREATER_THAN: return (sum_of_weights > threshold)
        else: raise ValueError("invalid threshold operator")

    def _get_split_constraint(self, tree, node):
        feat_id = self._trees.split_feat_id(tree, node)
        split_value = self._trees.split_value(tree, node)
        return (self._xvars[feat_id] < split_value)

    def _pop_best_tree(self, threshold_op):
        if threshold_op == LESS_THAN:
            self._remaining_trees.sort(key=lambda tree: self._lobounds[tree])
            #print(list(map(lambda tree: (tree, self._lobounds[tree]), self._remaining_trees)))
        else:
            self._remaining_trees.sort(key=lambda tree: self._upbounds[tree], reverse=True)
            #print(list(map(lambda tree: (tree, self._upbounds[tree]), self._remaining_trees)))
        return self._remaining_trees.pop()

    def _encode_tree(self, tree, node):
        if not self._trees.is_reachable(tree, node):
            return False

        wvar = self._wvars[tree]

        if self._trees.is_leaf(tree, node):
            return (wvar == self._trees.leaf_value(tree, node))
        else:
            l, r = self._trees.left(tree, node), self._trees.right(tree, node)
            lc = self._encode_tree(tree, l)
            rc = self._encode_tree(tree, r)
            c = self._get_split_constraint(tree, node)

            if lc == False and rc == False:
                return False
            elif lc == False:
                return z3.And(z3.Not(c), rc)
            elif rc == False:
                return z3.And(c, lc)
            else:
                return z3.Or(z3.And(c, lc), z3.And(z3.Not(c), rc))

    def _encode_tree_bound(self, tree):
        wvar = self._wvars[tree]
        lo = self._lobounds[tree]
        hi = self._upbounds[tree]
        if math.isinf(lo) or math.isinf(hi):
            return False
        return z3.And((lo <= wvar), (wvar <= hi))

    def _solver_check(self, constraint=None):
        if constraint is None:
            return self._solver.check() == z3.sat
        else:
            status = self._solver.check(constraint) == z3.sat
            return status
