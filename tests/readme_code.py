### <PART example_at>
import numpy as np
from veritas import *

# Manually create a two-tree ensemble
#
#       F0 < 2                     F0 < 3
#       /    \                     /    \         
#   F0 < 1   F0 < 3     +     F1 < 5     F0 < 0
#   /   \     /   \           /   \       /   \  
#  1     2   3     4         10   20     30   F2 < 1
#                                             /    \
#                                            50    60
at = AddTree()
t = at.add_tree();
t.split(t.root(), 0, 2)   # split(node_id, feature_id, split_value)
t.split( t.left(t.root()), 0, 1)
t.split(t.right(t.root()), 0, 3)
t.set_leaf_value( t.left( t.left(t.root())), 3)
t.set_leaf_value(t.right( t.left(t.root())), 4)
t.set_leaf_value( t.left(t.right(t.root())), 5)
t.set_leaf_value(t.right(t.right(t.root())), 6)

t = at.add_tree();
t.split(t.root(), 0, 3)
t.split( t.left(t.root()), 1, 5)
t.split(t.right(t.root()), 1, 0)
t.split(t.right(t.right(t.root())), 2) # Boolean split (ie < 1.0)
t.set_leaf_value( t.left( t.left(t.root())), 30)
t.set_leaf_value(t.right( t.left(t.root())), 40)
t.set_leaf_value( t.left(t.right(t.root())), 50)
t.set_leaf_value( t.left(t.right(t.right(t.root()))), 70)
t.set_leaf_value(t.right(t.right(t.right(t.root()))), 80)

# Print the trees (including the node-ids)
print(at[0])
print(at[1])

# Evaluate this ensemble
print("Eval:", at.eval(np.array([[0, 0, 0], [15, -3, 9]])))
### </PART>


print("---------------\n")
### <PART max_output>
# What is the maximum of the ensemble?
s = Search.max_output(at)
s.steps(100)

print("Global maximum")
if s.num_solutions() > 0:
    sol = s.get_solution(0)
    print("- current best solution:", sol.output, "->",
          "optimal" if s.is_optimal() else "suboptimal", "solution")
    print("- feature value ranges", sol.box())
    sol_nodes = s.get_solution_nodes(0)
    print("  which lead to leaf nodes", sol_nodes,
          "with leaf values",
          [at[i].get_leaf_value(n) for i, n in enumerate(sol_nodes)])

### </PART>


print("---------------\n")
### <PART min_output_constrained>
# If feature0 is between 3 and 5, what is the minimum possible output?
prune_box = [(0, Domain(3, 5))]  # (feat_id, domain) list, sorted by feat_id
at_neg = at.negate_leaf_values() # maximize with -leaf_values == minimize
s = Search.max_output(at_neg)
s.prune(prune_box)
s.steps(100)

print("Minimum with feature0 in [3, 5]")
if s.num_solutions() > 0:
    sol = s.get_solution(0)
    print("- current best solution:", -sol.output, "->",
          "optimal" if s.is_optimal() else "suboptimal", "solution")
    print("- feature value ranges", sol.box())
    sol_nodes = s.get_solution_nodes(0)
    print("  which lead to leaf nodes", sol_nodes,
          "with leaf values",
          [at[i].get_leaf_value(n) for i, n in enumerate(sol_nodes)])
### </PART>

print("---------------\n")
### <PART featmap>
# For two instances X0 and X1, allowing only feature3 to be different between
# the two instances, what is the maximum output difference at(X1)-at(X0)?
feat_map = FeatMap(["feature1", "feature2", "feature3"])
feat_map.use_same_id_for(feat_map.get_index("feature1", 0),
                         feat_map.get_index("feature1", 1))
feat_map.use_same_id_for(feat_map.get_index("feature2", 0),
                         feat_map.get_index("feature2", 1))

# `at_renamed` will use a different id for feature3, but the same id for
# feature0 and feature1
print("feat_id used for feature3 for instances:",
        feat_map.get_feat_id("feature3", 0),
        feat_map.get_feat_id("feature3", 1))
at_contrast = at.concat_negated(feat_map.transform(at, 1))

print()
print(at_contrast[1])
print(at_contrast[3])
### </PART>

### <PART print_featmap>
print(feat_map)
### </PART>

print("\n---------------\n")

### <PART two_instances>
s = Search.max_output(at_contrast)
s.step_for(10.0, 10)

print("Maximum difference between instance0 and instance1")
if s.num_solutions() > 0:
    sol = s.get_solution(0)
    print("- current best solution:", sol.output, "->",
          "optimal" if s.is_optimal() else "suboptimal", "solution")
    print("- feature value ranges", sol.box())
    sol_nodes = s.get_solution_nodes(0)
    print("  which lead to leaf nodes", sol_nodes,
          "with leaf values",
          [at[i].get_leaf_value(n) for i, n in enumerate(sol_nodes[0:2])],
          [at[i].get_leaf_value(n) for i, n in enumerate(sol_nodes[2:4])])
### </PART>



print("\n---------------\n")

### <PART robustness0>
# Checking robustness
# We change the `base_score` of the ensemble so that we can have negative
# outputs, which is necessary for robustness checking (we want classes to
# flip!)
at.base_score = -44

# Generate all possible output configurations for this `at`
s = Search.max_output(at)
done = s.steps(100)
while not done:
    done = s.steps(100)

print("{:<3} {:<10} {}".format("i", "output", "box"))
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(f"{i:<3} {sol.output:<10} {sol.box()}")
### </PART>

### <PART robustness0_eval>
example = [2, 4, 2]
print("output for example", example, "is", at.eval(example)[0])
### </PART>

### <PART robustness1>
from veritas import VeritasRobustnessSearch
rob = VeritasRobustnessSearch(None, at, example, start_delta=5.0)
delta, delta_lo, delta_up = rob.search()

print("adversarial examples:", rob.generated_examples,
        "with outputs", at.eval(np.array(rob.generated_examples)))
### </part>


# We can verify this result using the MILP approach (Kantchelian et al.'16):
### <part robustness1_kan>
from veritas.kantchelian import KantchelianAttack
kan = KantchelianAttack(at, target_output=True, example=example, silent=True)
kan.optimize()
adv_example, adv_output = kan.solution()[:2]
print("Kantchelian adversarial example", adv_example, "with output", adv_output)
### </part>




print("\n---------------\n")

### <part onehot0>
# Constraints: one-hot (feature0 and feature1 cannot be true at the same time)
# That is, the model below can only output 0: -100 + 100 and 100 - 100
at = AddTree()
t = at.add_tree();
t.split(t.root(), 0)   # Boolean split(node_id, feature_id, split_value)
t.set_leaf_value( t.left(t.root()), -100)
t.set_leaf_value(t.right(t.root()), 100)

t = at.add_tree();
t.split(t.root(), 1)   # Boolean split(node_id, feature_id, split_value)
t.set_leaf_value( t.left(t.root()), -100)
t.set_leaf_value(t.right(t.root()), 100)

print(at[0])
print(at[1])

# Without one-hot constraint: solution is incorrect feat0 == true && feat1 ==
# true leading to output of 200.
s = Search.max_output(at)
s.steps(100)
print("\nWithout one-hot constraint")
print("{:<3} {:<10} {}".format("i", "output", "box"))
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(f"{i:<3} {sol.output:<10} {sol.box()}")
print("number of rejected states due to constraint:", s.num_rejected_states)
### </part>

### <part onehot1>

# With constraint:
s = Search.max_output(at)
s.add_onehot_constraint([0, 1])
s.steps(100)
print("\nWith one-hot constraint")
print("{:<3} {:<10} {}".format("i", "output", "box"))
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(f"{i:<3} {sol.output:<10} {sol.box()}")
print("number of rejected states due to constraint:", s.num_rejected_states)
### </part>
