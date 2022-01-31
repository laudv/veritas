## \file kantchelian.py
#
# Inspired by Chen Hongge's implementation:
#    https://github.com/chenhongge/RobustTrees/blob/ed28228ab68e2c9f0fe630c7a7faa70e8411a359/xgbKantchelianAttack.py
#
# > Hongge Chen, Huan Zhang, Duane Boning, and Cho-Jui Hsieh "Robust Decision
# > Trees Against Adversarial Examples", ICML 2019
#
# Algorithm from
#
# > Kantchelian, Alex, J. Doug Tygar, and Anthony Joseph. "Evasion and
# > hardening of tree ensemble classifiers." International Conference on Machine
# > Learning. 2016.
#
# This requires `gurobipy` to be installed: https://www.gurobi.com
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import timeit, time
import gurobipy as gu
import numpy as np
from veritas import AddTree, Domain

DUMMY_AT = AddTree()

class NodeInfo:
    def __init__(self, var, leafs_in_subtree):
        self.leafs_in_subtree = leafs_in_subtree
        self.var = var

## \ingroup python
# \brief Base class for MILP methods
class KantchelianBase:

    def __init__(self, split_values, max_time=1e100):
        self.guard = 1e-4
        self.split_values = split_values
        self.env = gu.Env(empty=True)
        self.env.setParam("Threads", 1)
        try:
            self.model = gu.Model("KantchelianAttack", env=self.env) # requires license
        except:
            print("Gurobi: could not set env, using defaults")
            self.model = gu.Model("KantchelianAttack")#, env=self.env) # requires license
        self.pvars = self._construct_pvars()

        self.max_time = max_time
        self.total_time = None
        self.total_time_p = None
        self.bounds = []
        self.times = []
        self.force_stop = False

    def stats(self):
        return {
            "bounds": self.bounds,
            "times": self.times,
            "max_time": self.max_time,
            "time": self.total_time,
            "time_p": self.total_time_p,
            "force_stop": self.force_stop
        }

    def optimize(self):
        self.start_time = timeit.default_timer()
        self.start_time_p = time.process_time()
        self.model.optimize(self._optimize_callback())
        up = self.model.getAttr(gu.GRB.Attr.ObjBound)
        lo = self.model.getAttr(gu.GRB.Attr.ObjVal)
        self.bounds.append((lo, up))
        self.total_time = timeit.default_timer() - self.start_time
        self.total_time_p = time.process_time() - self.start_time_p
        self.times.append((self.total_time, self.total_time_p))

    def constrain_to_box(self, box):
        for attribute, dom in enumerate(box):
            lo, hi = dom.lo, dom.hi
            if attribute not in self.split_values:
                continue
            split_values = self.split_values[attribute]
            #print(lo, hi)
            for val in split_values:
                var = self.pvars[(attribute, val)]
                if val <= lo:
                    self.model.addConstr(var == 0)
                if val >= hi:
                    self.model.addConstr(var == 1)

    def _check_time(self, model):
        if self.start_time_p + self.max_time < time.process_time():
            print(f"Terminating Gurobi after {self.max_time} processor seconds")
            self.force_stop = True
            model.terminate()

    def _optimize_callback_fn(self, model, where):
        if where == gu.GRB.Callback.MIP:
            t = model.cbGet(gu.GRB.Callback.RUNTIME)
            t_p = time.process_time() - self.start_time_p
            self.times.append((t, t_p))
            lo = model.cbGet(gu.GRB.Callback.MIP_OBJBST)
            up = model.cbGet(gu.GRB.Callback.MIP_OBJBND)
            self.bounds.append((lo, up))
            self._check_time(model)

        if where == gu.GRB.Callback.MIPSOL:
            t = model.cbGet(gu.GRB.Callback.RUNTIME)
            t_p = time.process_time() - self.start_time_p
            self.times.append((t, t_p))
            lo = model.cbGet(gu.GRB.Callback.MIPSOL_OBJBST)
            up = model.cbGet(gu.GRB.Callback.MIPSOL_OBJBND)
            self.bounds.append((lo, up))
            self._check_time(model)

        if where == gu.GRB.Callback.MIPNODE:
            t = model.cbGet(gu.GRB.Callback.RUNTIME)
            t_p = time.process_time() - self.start_time_p
            self.times.append((t, t_p))
            lo = model.cbGet(gu.GRB.Callback.MIPNODE_OBJBST)
            up = model.cbGet(gu.GRB.Callback.MIPNODE_OBJBND)
            self.bounds.append((lo, up))
            self._check_time(model)

    def _optimize_callback(self):
        return lambda m, w: self._optimize_callback_fn(m, w)

    def _construct_pvars(self): # uses self.split_values, self.model
        pvars = {}
        for attribute, split_values in self.split_values.items():
            for k, split_value in enumerate(split_values): # split values are sorted
                var = self.model.addVar(vtype=gu.GRB.BINARY, name=f"p{attribute}-{k}")
                pvars[(attribute, split_value)] = var
        return pvars

    def _collect_node_info(self, at): # uses self.model, self.pvars
        node_info_per_tree = []
        for tree_index in range(len(at)):
            tree = at[tree_index]
            leafs_of_node = {}
            var_of_node = {}

            def traverse(node):
                if node in leafs_of_node:
                    return leafs_of_node[node]
                if tree.is_leaf(node):
                    var = self.model.addVar(lb=0.0, ub=1.0,
                            name=f"l{tree_index}-{node}")
                    leafs = [(node)]
                else:
                    split = tree.get_split(node)
                    var = self.pvars[(split.feat_id, split.split_value)]
                    leafs = traverse(tree.left(node)) \
                            + traverse(tree.right(node))
                leafs_of_node[node] = leafs
                var_of_node[node] = var
                return leafs
            traverse(tree.root())

            node_infos = {}
            for node in sorted(leafs_of_node.keys()): # sorted by node_id
                node_infos[node] = NodeInfo(var_of_node[node],
                        leafs_of_node[node])
            node_info_per_tree.append(node_infos)
        return node_info_per_tree

    def _add_leaf_consistency(self, at, node_info_per_tree): # uses self.model
        for tree_index, node_infos in enumerate(node_info_per_tree):
            root_node_info = node_infos[at[tree_index].root()]
            vars = [node_infos[l].var for l in root_node_info.leafs_in_subtree]
            coef = [1] * len(vars)

            self.model.addConstr(gu.LinExpr(coef, vars) == 1.0,
                    name=f"leaf_sum{tree_index}")

    def _add_predicate_leaf_consistency(self, at, node_info_per_tree): # uses self.model
        for tree_index in range(len(at)):
            tree = at[tree_index]
            node_infos = node_info_per_tree[tree_index]

            stack = [tree.root()]
            while len(stack) > 0:
                node = stack.pop()
                if tree.is_leaf(node): continue
                pvar = node_infos[node].var
                left, right = tree.left(node), tree.right(node)
                left_leafs = node_infos[left].leafs_in_subtree
                right_leafs = node_infos[right].leafs_in_subtree
                left_lvars = [node_infos[n].var for n in left_leafs]
                right_lvars = [node_infos[n].var for n in right_leafs]

                left_sum = gu.LinExpr([1]*len(left_lvars), left_lvars)
                right_sum = gu.LinExpr([1]*len(right_lvars), right_lvars)

                if tree.is_root(node):
                    # if pvar is true, then the right branch cannot contain a true leaf
                    self.model.addConstr(right_sum+pvar == 1.0, name=f"pl_consist_r{node}")
                    # if pvar is false, then left sum cannot contain a true leaf
                    self.model.addConstr(left_sum-pvar == 0.0, name=f"pl_consist_l{node}")
                else:
                    # if pvar is true, right cannot have a true leaf
                    self.model.addConstr(right_sum+pvar <= 1.0, name=f"pl_consist_r{node}")
                    # if pvar is false, left cannot have a true leaf
                    self.model.addConstr(left_sum-pvar <= 0.0, name=f"pl_consist_l{node}")

                stack += [right, left]

    def _add_predicate_consistency(self): # uses self.split_values, self.pvars, self.model
        for attribute, split_values in self.split_values.items():
            # predicate in split is X < split_value
            # split values are sorted
            var0 = self.pvars[(attribute, split_values[0])]
            for k, split_value1 in enumerate(split_values[1:]):
                var1 = self.pvars[(attribute, split_value1)]
                self.model.addConstr(var0 <= var1, f"p_consist{k}")
                var0 = var1

    def _add_mislabel_constraint(self, at, node_info_per_tree, target_output): # uses self.model
        ensemble_output = self._get_ensemble_output_expr(at, node_info_per_tree)
        if target_output: # positive class
            self.model.addConstr(ensemble_output >= 0.0, name=f"mislabel_pos")
        else: # negative class
            self.model.addConstr(ensemble_output <= 0.0, name=f"mislabel_neg")

    def _get_ensemble_output_expr(self, at, node_info_per_tree):
        leaf_values = []
        vars = []

        for tree_index in range(len(at)):
            tree = at[tree_index]
            node_infos = node_info_per_tree[tree_index]
            leafs = node_infos[tree.root()].leafs_in_subtree
            vars += [node_infos[n].var for n in leafs]
            leaf_values += [tree.get_leaf_value(n) for n in leafs]

        return (gu.LinExpr(leaf_values, vars) + at.base_score)

    def _add_robustness_objective(self, example): # uses self.model, split_values, pvars, adds self.bvar
        self.bvar = self.model.addVar(name="b")
        
        for attribute, split_values in self.split_values.items():
            pvars = [self.pvars[(attribute, split_value)] for split_value in split_values]
            x = self.example[attribute]
            w = self._get_objective_weights(split_values, x)

            expr = gu.LinExpr(w[:-1], pvars) + w[-1]
            self.model.addConstr(expr <= self.bvar, name=f"obj{attribute}")

        self.model.setObjective(self.bvar, gu.GRB.MINIMIZE)

    def _add_output_objective(self, at, node_info_per_tree, sense=gu.GRB.MAXIMIZE):
        output = self._get_ensemble_output_expr(at, node_info_per_tree)
        self.model.setObjective(output, sense)

    def _get_objective_weights(self, split_values, x):
        axis = [-np.inf] + split_values + [np.inf]
        w = [0.0] * (len(split_values)+1)
        for k in range(len(split_values)+1, 0, -1):
            tau1 = axis[k]
            tau0 = axis[k-1]
            if tau0 <= x and x < tau1: # from paper: j == k-1 => interval of x
                w[k-1] = 0.0
            elif x < tau0 and x < tau1:
                w[k-1] = np.abs(x - tau0)
            elif x >= tau0 and x >= tau1:
                w[k-1] = np.abs(x - tau1 + self.guard)
            else:
                assert False
        for k in range(len(w)-1):
            w[k] -= w[k+1]
        return w

    def has_solution(self):
        return self.model.status == gu.GRB.OPTIMAL \
                or self.model.status == gu.GRB.SUBOPTIMAL

    def objective_bound(self):
        return self.model.objBound;

    def _extract_adv_example(self, example): # uses self.split_values, self.pvars, self.guard
        adv_example = example.copy()
        for attribute, split_values in self.split_values.items():
            pvars = [self.pvars[(attribute, split_value)] for split_value in split_values]
            x = self.example[attribute]
            #print("solution", attribute, x, [(p.x, s) for p, s in zip(pvars, split_values)])
            for split_value in split_values:
                pvar = self.pvars[(attribute, split_value)]
                if pvar.x > 0.5 and x >= split_value: # greater than or equal to split_value
                    #print("adjusting attribute", attribute, "down from", x, "to", split_value-self.guard)
                    adv_example[attribute] = split_value - self.guard
                    break # pick first active pvar we encounter
                if pvar.x <= 0.5 and x < split_value:
                    #print("adjusting attribute", attribute, "up from", x, "to", split_value)
                    adv_example[attribute] = split_value# + self.guard
                    # pick last active pvar we encouter!
        return adv_example

    def _extract_intervals(self):
        # (!) these intervals can be more specific than strictly necessary
        # because it assigns values to pvars even when they don't occur in the
        # taken paths
        intervals = {}
        for attribute, split_values in self.split_values.items():
            pvars = [self.pvars[(attribute, split_value)] for split_value in split_values]
            lo, hi = -np.inf, np.inf
            for split_value in split_values:
                # pvar true means go left (ie less than split value)
                pvar = self.pvars[(attribute, split_value)]
                if pvar.x > 0.5 and hi == np.inf: # greater than or equal to split_value
                    #print("POS ", attribute, split_value)
                    hi = split_value # pick the first active pvar (lowest split value)
                if pvar.x <= 0.5:
                    #print("NEG ", attribute, split_value)
                    lo = split_value # pick last active pvar we encouter!
            intervals[attribute] = Domain.exclusive(lo, hi)
        return {attr: intervals[attr] for attr in sorted(intervals)}

    def _extract_ensemble_output(self, at, node_info_per_tree):
        ensemble_output = at.base_score
        for tree_index in range(len(at)):
            tree = at[tree_index]
            node_infos = node_info_per_tree[tree_index]
            leafs = node_infos[tree.root()].leafs_in_subtree
            lvars = [node_infos[n].var for n in leafs]
            tree_output = sum(0.0 if lvar.x < 0.5 else tree.get_leaf_value(n)
                    for lvar, n in zip(lvars, leafs))
            ensemble_output += tree_output
        return ensemble_output



## \ingroup python
# \brief Robustness checking with MILP
class KantchelianAttack(KantchelianBase):

    def __init__(self, at, target_output, example, **kwargs):
        self.at = at
        self.example = example

        super().__init__(self.at.get_splits(), **kwargs)

        self.node_info_per_tree = self._collect_node_info(self.at)

        # debug print [tree_index, node_id, gurobi var, leafs in node's subtree]
        #for tree_index, node_infos in enumerate(self.node_info_per_tree):
        #    for node, node_info in node_infos.items():
        #        print(tree_index, node, node_info.var, node_info.leafs_in_subtree)

        # CONSTRAINT: exactly one leaf is active per tree
        self._add_leaf_consistency(self.at, self.node_info_per_tree)

        # CONSTRAINT: If pvar of a predicate is true, then false branch cannot
        # have a true leaf, and vice versa. This is strict for the root.
        self._add_predicate_leaf_consistency(self.at, self.node_info_per_tree)

        # CONSTRAINT: Predicate consistency: if X < 5 is true, then X < 7 must
        # also be true.
        self._add_predicate_consistency()

        # CONSTRAINT: mislabel constraint: instance we find should have a
        # specific class
        self._add_mislabel_constraint(self.at, self.node_info_per_tree,
                target_output=target_output)

        self._add_robustness_objective(self.example)

        self.model.update()
        #print(self.model.display())

    def solution(self):
        adv_example = self._extract_adv_example(self.example)
        ensemble_output = self._extract_ensemble_output(self.at,
                self.node_info_per_tree)
        return adv_example, ensemble_output, self.bvar.x

## \ingroup python
# \brief Targeted robustness checking with MILP
class KantchelianTargetedAttack(KantchelianBase):

    def __init__(self, source_at, target_at, example, **kwargs):
        self.source_at = source_at if source_at is not None else DUMMY_AT
        self.target_at = target_at if target_at is not None else DUMMY_AT
        self.example = example

        super().__init__(self._combine_split_values(), **kwargs)

        self.source_node_info_per_tree = self._collect_node_info(self.source_at)
        self.target_node_info_per_tree = self._collect_node_info(self.target_at)

        self._add_leaf_consistency(self.source_at, self.source_node_info_per_tree)
        self._add_leaf_consistency(self.target_at, self.target_node_info_per_tree)

        self._add_predicate_leaf_consistency(self.source_at, self.source_node_info_per_tree)
        self._add_predicate_leaf_consistency(self.target_at, self.target_node_info_per_tree)

        self._add_predicate_consistency()

        ## source_at < 0 && target_at > 0
        #self._add_mislabel_constraint(self.source_at,
        #        self.source_node_info_per_tree, target_output=False)
        #self._add_mislabel_constraint(self.target_at,
        #        self.target_node_info_per_tree, target_output=True)

        # same mislabel constraint as Veritas/Merge: more confident about
        # target than source
        self._add_multiclass_mislabel_constraint()

        self._add_robustness_objective(self.example)

        self.model.update()

    def solution(self):
        adv_example = self._extract_adv_example(self.example)
        ensemble_output0 = self._extract_ensemble_output(self.source_at,
                self.source_node_info_per_tree)
        ensemble_output1 = self._extract_ensemble_output(self.target_at,
                self.target_node_info_per_tree)
        return adv_example, ensemble_output0, ensemble_output1, self.bvar.x

    def _combine_split_values(self):
        source_split_values = self.source_at.get_splits()
        target_split_values = self.target_at.get_splits()

        split_values = {}
        attributes = source_split_values.keys() | target_split_values.keys()
        for attr in attributes:
            split_values[attr] = sorted(source_split_values.get(attr, []) \
                + target_split_values.get(attr, []))

        return split_values

    def _add_multiclass_mislabel_constraint(self):
        source_ensemble_output = self._get_ensemble_output_expr(self.source_at,
                self.source_node_info_per_tree)
        target_ensemble_output = self._get_ensemble_output_expr(self.target_at,
                self.target_node_info_per_tree)

        self.model.addConstr(source_ensemble_output <= target_ensemble_output,
                name="multiclass_mislabel")


## \ingroup python
# \brief Variation of the Kantchelian attack where the output is optimized rather than
# the distance to the closest adversarial example.
class KantchelianOutputOpt(KantchelianBase):
    def __init__(self, at, **kwargs):
        self.at = at
        super().__init__(self.at.get_splits(), **kwargs)
        self.node_info_per_tree = self._collect_node_info(self.at)
        self._add_leaf_consistency(self.at, self.node_info_per_tree)
        self._add_predicate_leaf_consistency(self.at, self.node_info_per_tree)
        self._add_predicate_consistency()

        self._add_output_objective(self.at, self.node_info_per_tree, sense=gu.GRB.MAXIMIZE)
        # self._add_output_objective(self.at, self.node_info_per_tree, sense=gu.GRB.MINIMIZE)

        self.model.update()

    def solution(self):
        intervals = self._extract_intervals()
        return self.model.objVal, intervals
