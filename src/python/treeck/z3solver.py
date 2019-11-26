import math
import z3

from .consts import LESS_THAN, GREATER_THAN

class Z3Solver:

    def __init__(self, domains, addtree):
        self._num_features = len(domains)
        self._domains = domains
        self._addtree = addtree
        self._timeout = 4294967295

        self._xvars = [z3.Real("x{}".format(i)) for i in range(self._num_features)]
        self._wvars = [z3.Real("w{}".format(i)) for i in range(len(self._addtree))]

    def xvar(self, i):
        return self._xvars[i]

    def set_timeout(self, timeout):
        self._timeout = timeout

    def verify(self, constraints=[], threshold=0.0, op=LESS_THAN):
        self._remaining_trees = list(range(len(self._addtree)))
        self._lobounds = [ math.inf] * len(self._addtree)
        self._upbounds = [-math.inf] * len(self._addtree)
        self._unreachable = set()
        self._iteration_count = 0

        self._solver = z3.Solver()
        self._solver.set("timeout", self._timeout)
        self._solver.add(*constraints)
        self._solver.add(*self._domain_constraints())
        self._solver.add(self._output_constraint(threshold, op))

        status = z3.sat

        while len(self._remaining_trees) > 0 and status == z3.sat:
            self._iteration_count += 1

            # Compute bounds and reachable branches for all remaining trees
            for tree in self._remaining_trees:
                bounds_changed = self._test_tree_reachability(tree)
                if bounds_changed:
                    self._solver.add(self._encode_tree_bound(tree))

            # Add the tree with the best bound
            best_tree_index = self._pop_best_tree(op)
            best_tree = self._addtree[best_tree_index]
            enc = self._encode_tree(best_tree, best_tree.root())
            self._solver.add(enc)

            status = self._solver.check()
            #print("{:4}: DEBUG added tree{} with bounds [{:.4g}, {:.4g}] -> {}\n".format(
            #            self._iteration_count, best_tree_index,
            #            self._lobounds[best_tree_index],
            #            self._upbounds[best_tree_index], status))

        #print(self._solver.to_smt2())
        #print("status:", status)
        return status

    def model(self):
        z3model = self._solver.model()
        xs = []
        for x in self._xvars:
            xs.append(self._extract_var(z3model, x))
        ws = []
        for w in self._wvars:
            val = self._extract_var(z3model, w)
            assert val is not None
            ws.append(val)
        return { "xs": xs, "ws": ws }

    def _test_tree_reachability(self, tree_index):
        tree = self._addtree[tree_index]
        stack = [(tree.root(), True)]

        lo =  math.inf
        hi = -math.inf

        while len(stack) > 0:
            node, path_constraint = stack.pop()

            if not self._is_reachable(tree, node): continue

            if tree.is_leaf(node):
                leaf_value = tree.get_leaf_value(node)
                lo = min(lo, leaf_value)
                hi = max(hi, leaf_value)
                continue

            split = self._get_split_constraint(tree, node)
            lpath_constraint = z3.And(path_constraint, split)
            rpath_constraint = z3.And(path_constraint, z3.Not(split))

            l, r = tree.left(node), tree.right(node)
            lreachable = self._is_reachable(tree, l)
            rreachable = self._is_reachable(tree, r)

            # RIGHT
            if rreachable:
                if self._solver_check(rpath_constraint):
                    stack.append((r, rpath_constraint))
                else:
                    #print("{:4}: DEBUG unreachable right tree{}: {}->{}".format(
                    #    self._iteration_count, tree.index(), node, r))
                    self._mark_unreachable(tree, r)
                    assert not self._is_reachable(tree, r)

            # LEFT
            if lreachable:
                if self._solver_check(lpath_constraint):
                    stack.append((l, lpath_constraint))
                else:
                    #print("{:4}: DEBUG unreachable left tree{}: {}->{}".format(
                    #    self._iteration_count, tree.index(), node, l))
                    self._mark_unreachable(tree, l)
                    assert not self._is_reachable(tree, l)

        changed = self._lobounds[tree_index] != lo \
                or self._upbounds[tree_index] != hi

        self._lobounds[tree_index] = lo
        self._upbounds[tree_index] = hi

        #if changed:
        #    print("{:4}: DEBUG new bounds tree{}: [{:.4g}, {:.4g}]".format(
        #        self._iteration_count, tree.index(), lo, hi))
        return changed

    def _is_reachable(self, tree, node):
        return (tree.index(), node) not in self._unreachable

    def _mark_unreachable(self, tree, node):
        return self._unreachable.add((tree.index(), node))

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
        sum_of_weights = self._addtree.base_score
        for i in range(len(self._wvars)):
            sum_of_weights = sum_of_weights + self._wvars[i]
        if   op == LESS_THAN:    return (sum_of_weights < threshold)
        elif op == GREATER_THAN: return (sum_of_weights > threshold)
        else: raise ValueError("invalid threshold operator")

    def _get_split_constraint(self, tree, node):
        feat_id, split_value = tree.get_split(node)
        return (self._xvars[feat_id] < split_value)

    def _pop_best_tree(self, threshold_op):
        if threshold_op == LESS_THAN:
            self._remaining_trees.sort(key=lambda i: self._lobounds[i])
        else:
            self._remaining_trees.sort(key=lambda i: self._upbounds[i], reverse=True)
        return self._remaining_trees.pop()

    def _encode_tree(self, tree, node):
        if not self._is_reachable(tree, node):
            return False

        wvar = self._wvars[tree.index()]

        if tree.is_leaf(node):
            return (wvar == tree.get_leaf_value(node))
        else:
            l, r = tree.left(node), tree.right(node)
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

    def _extract_var(self, z3model, var):
        val = z3model[var]
        if val is None:
            return None
        if z3.is_rational_value(val):
            n = float(val.numerator_as_long())
            d = float(val.denominator_as_long())
            return float(n / d)
        raise RuntimeError("var not supported")
