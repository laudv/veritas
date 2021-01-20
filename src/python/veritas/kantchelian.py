## Inspired by:
##   https://github.com/chenhongge/RobustTrees/blob/ed28228ab68e2c9f0fe630c7a7faa70e8411a359/xgbKantchelianAttack.py
##
## Algorithm from
##    Kantchelian, Alex, J. Doug Tygar, and Anthony Joseph. "Evasion and
##    hardening of tree ensemble classifiers." International Conference on Machine
##    Learning. 2016.

import gurobipy as gu
import numpy as np

class NodeInfo:
    def __init__(self, var, leafs_in_subtree):
        self.leafs_in_subtree = leafs_in_subtree
        self.var = var

class MILP:

    def __init__(self, at, target_label, example):
        self.at = at
        self.target_label = target_label
        self.example = example
        self.split_values = self.at.get_splits()
        self.model = gu.Model("milp")
        self.pvars = {}

        self.guard = 1e-4

        for attribute, split_values in self.split_values.items():
            for k, split_value in enumerate(split_values): # split values are sorted
                var = self.model.addVar(vtype=gu.GRB.BINARY, name=f"p{attribute}-{k}")
                self.pvars[(attribute, split_value)] = var

        self._collect_node_info()
        self.model.update()

        # debug print [tree_index, node_id, gurobi var, leafs in node's subtree]
        #for tree_index, node_infos in enumerate(self.node_info_per_tree):
        #    for node, node_info in node_infos.items():
        #        print(tree_index, node, node_info.var, node_info.leafs_in_subtree)

        # CONSTRAINT: exactly one leaf is active per tree
        self._add_leaf_consistency()

        # CONSTRAINT: If pvar of a predicate is true, then false branch cannot
        # have a true leaf, and vice versa. This is strict for the root.
        self._add_predicate_leaf_consistency()

        # CONSTRAINT: Predicate consistency: if X < 5 is true, then X < 7 must
        # also be true.
        self._add_predicate_consistency()

        # CONSTRAINT: mislabel constraint: instance we find should have a
        # specific class
        self._add_mislabel_constraint(target_label=target_label)

        # OBJECTIVE: infinity norm only
        self._add_objective()

        self.model.update()
        #print(self.model.display())

    def optimize(self):
        self.model.optimize()
        #for v in self.model.getVars():
        #    print(f"{v.varName} {v.x}")

    def solution(self):
        adv_example = self.example.copy()
        for attribute, split_values in self.split_values.items():
            pvars = [self.pvars[(attribute, split_value)] for split_value in split_values]
            x = self.example[attribute]
            #print("solution", attribute, x, [(p.x, s) for p, s in zip(pvars, split_values)])
            for split_value in split_values:
                pvar = self.pvars[(attribute, split_value)]
                if pvar.x > 0.5 and x >= split_value: # greater than or equal to split_value
                    print("adjusting attribute", attribute, "down from", x, "to", split_value-self.guard)
                    adv_example[attribute] = split_value - self.guard
                if pvar.x <= 0.5 and x < split_value:
                    print("adjusting attribute", attribute, "up from", x, "to", split_value)
                    adv_example[attribute] = split_value# + self.guard

        ensemble_output = 0.0
        for tree_index in range(len(self.at)):
            tree = self.at[tree_index]
            node_infos = self.node_info_per_tree[tree_index]
            leafs = node_infos[tree.root()].leafs_in_subtree
            lvars = [node_infos[n].var for n in leafs]
            tree_output = sum(0.0 if lvar.x < 0.5 else tree.get_leaf_value(n)
                    for lvar, n in zip(lvars, leafs))
            ensemble_output += tree_output

        return adv_example, ensemble_output, self.bvar.x

    def _collect_node_info(self):
        self.node_info_per_tree = []
        for tree_index in range(len(self.at)):
            tree = self.at[tree_index]
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
            self.node_info_per_tree.append(node_infos)

    def _add_leaf_consistency(self):
        for tree_index, node_infos in enumerate(self.node_info_per_tree):
            root_node_info = node_infos[self.at[tree_index].root()]
            vars = [node_infos[l].var for l in root_node_info.leafs_in_subtree]
            coef = [1] * len(vars)

            self.model.addConstr(gu.LinExpr(coef, vars) == 1.0,
                    name=f"leaf_sum{tree_index}")

    def _add_predicate_leaf_consistency(self):
        for tree_index in range(len(self.at)):
            tree = self.at[tree_index]
            node_infos = self.node_info_per_tree[tree_index]

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

    def _add_predicate_consistency(self):
        for attribute, split_values in self.split_values.items():
            # predicate in split is X < split_value
            # split values are sorted
            var0 = self.pvars[(attribute, split_values[0])]
            for k, split_value1 in enumerate(split_values[1:]):
                var1 = self.pvars[(attribute, split_value1)]
                self.model.addConstr(var0 <= var1, f"p_consist{k}")
                var0 = var1

    def _add_mislabel_constraint(self, target_label):
        leaf_values = []
        vars = []

        for tree_index in range(len(self.at)):
            tree = self.at[tree_index]
            node_infos = self.node_info_per_tree[tree_index]
            leafs = node_infos[tree.root()].leafs_in_subtree
            vars += [node_infos[n].var for n in leafs]
            leaf_values += [tree.get_leaf_value(n) for n in leafs]

        ensemble_output = gu.LinExpr(leaf_values, vars)
        if target_label == "POSITIVE" or target_label == 1:
            self.model.addConstr(ensemble_output >= 0.0, name=f"mislabel_pos")
        else:
            self.model.addConstr(ensemble_output <= 0.0, name=f"mislabel_neg")

    def _add_objective(self):
        self.bvar = self.model.addVar(name="b")
        
        for attribute, split_values in self.split_values.items():
            pvars = [self.pvars[(attribute, split_value)] for split_value in split_values]
            x = self.example[attribute]
            w = self._get_objective_weights(split_values, x)

            expr = gu.LinExpr(w[:-1], pvars) + w[-1]
            self.model.addConstr(expr <= self.bvar, name=f"obj{attribute}")

        self.model.setObjective(self.bvar, gu.GRB.MINIMIZE)

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
