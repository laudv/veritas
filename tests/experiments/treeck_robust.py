import sys
from veritas import LtSplit
from veritas import RobustnessSearch

import treeck
from treeck.verifier import Verifier
from treeck.z3backend import Z3Backend as Backend

# copy trees between veritas and treeck formats
def _veritas_at_to_treeck_at(veritas_at):
    at0 = veritas_at
    at1 = treeck.AddTree()

    at1.base_score = at1.base_score

    for tree_index in range(len(at0)):
        tree0 = at0[tree_index]
        tree1 = at1.add_tree()
        stack = [(tree0.root(), tree1.root())]
        while len(stack) > 0:
            node0, node1 = stack.pop()
            if tree0.is_internal(node0):
                split = tree0.get_split(node0)
                if isinstance(split, LtSplit):
                    tree1.split(node1, split.feat_id, split.split_value)
                else:
                    tree1.split(node1, split.feat_id)
                stack.append((tree0.right(node0), tree1.right(node1)))
                stack.append((tree0.left(node0), tree1.left(node1)))
            else:
                tree1.set_leaf_value(node1, tree0.get_leaf_value(node0))

    return at1


class TreeckRobustnessSearch(RobustnessSearch):
    def __init__(self, source_at, target_at, example, **kwargs):
        super().__init__(example, **kwargs)

        self.source_at = _veritas_at_to_treeck_at(source_at)
        self.target_at = _veritas_at_to_treeck_at(target_at)

    def get_max_output_difference(self, delta):
        dt = treeck.DomTree([(self.source_at, {}), (self.target_at, {})])
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())
        v.add_all_trees()

        fids0 = set(v.instance(0).feat_ids())
        fids1 = set(v.instance(1).feat_ids())
        fids01 = fids0 & fids1

        for i in fids0:
            x = v.instance(0).xvar(i)
            pixel = self.example[i]
            v.add_constraint((x > max(0, pixel-delta)) & (x < min(255, pixel+delta)))
        for i in fids1:
            x = v.instance(1).xvar(i)
            pixel = self.example[i]
            v.add_constraint((x > max(0, pixel-delta)) & (x < min(255, pixel+delta)))

        for i in fids01:
            x0 = v.instance(0).xvar(i)
            x1 = v.instance(1).xvar(i)
            pixel = self.example[i]
            v.add_constraint(x0 == x1)

        v.add_constraint(v.instance(0).fvar() <= v.instance(1).fvar())
        res = v.check()

        # Actual output_difference does not really matter, just use values to
        # direct the search
        if res == Verifier.Result.SAT:
            generated_example = self.get_closest_example(v.model())
            return 1.0, [generated_example] # output_difference > 0 => an example exists
        elif res == Verifier.Result.UNKNOWN:
            return 1.0, [] # output_difference > 0 => an example may exist
        else: # UNSAT
            return -1.0, [] # no adv example can exist

    def get_closest_example(self, model):
        closest = self.example.copy()
        print("Treeck generated example:", model[0]["f"], model[1]["f"])
        for fid, value in model[0]["xs"].items():
            #print(f"changing {fid} from {self.example[fid]} -> {value}")
            closest[fid] = value
        for fid, value in model[1]["xs"].items():
            #print(f"changing {fid} from {self.example[fid]} -> {value}")
            closest[fid] = value
        return closest
