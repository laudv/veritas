### <PART example_at>
import numpy as np
from veritas import *

# Manually create a two-tree ensemble
#
#       F0 < 2                     F0 < 3
#       /    \                     /    \         
#   F0 < 1   F0 < 3     +     F1 < 5     F1 < 0
#   /   \     /   \           /   \       /    \
#  3     4   5     6         30   40     50     F2
#                                             /    \
#                                            70    80

at = AddTree(1)  # Empty ensemble with int his case 1 value in the leafs
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
print(f"{at}\n")
print(at[0])
print(at[1])
### </PART>


print("---------------\n")
### <PART get_addtree_example>
import veritas
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X,Y) = make_moons(100)

clf = RandomForestClassifier(
        max_depth=4,
        random_state=0,
        n_estimators=3)
clf.fit(X, Y)

# Convert the RandomForestClassifier model to a Veritas tree ensemble
at = veritas.get_addtree(clf)

print(at)
for tree in at:
    print(tree)
### </PART>


print("---------------\n")
### <PART AddTreeConverter>
# Trivial Tree representation
#
#         F0             F0        
#       /    \    +    /    \              
#      10    20       12    13

myModel = [[10,20,0],[12,13,0]] # [left leaf, Right leaf, Boolean Feature]

class MyAddTreeConverter(AddTreeConverter):
    def get_addtree(self,model):
        # Implement AddTreeConverter using your own model
        addtree = AddTree(1)
        
        for tree in model:
            t = addtree.add_tree()
            t.split(t.root(),1)
            t.set_leaf_value(t.left(t.root()),tree[0])
            t.set_leaf_value(t.right(t.root()),tree[1])

        return addtree


# Add converter instance to the converter_manager 
add_addtree_converter(MyAddTreeConverter())

# Use get_addtree() on your own models
addtree = get_addtree(myModel)

print(f"{addtree}\n")

print(addtree[0])
print(addtree[1])
### </PART>


print("---------------\n")
### <PART max_output>
# What is the maximum of the ensemble?
config = Config(HeuristicType.MAX_OUTPUT)
s = config.get_search(at,{})

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
          [at[i].get_leaf_value(n,0) for i, n in enumerate(sol_nodes)])
### </PART>


print("---------------\n")
### <PART min_output_constrained>
# If feature0 is between 3 and 5, what is the minimum possible output?
prune_box = [(0, Interval(3, 5))]  # (feat_id, domain) list, sorted by feat_id

config = Config(HeuristicType.MIN_OUTPUT)
s = config.get_search(at,prune_box)

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
          [at[i].get_leaf_value(n,0) for i, n in enumerate(sol_nodes)])
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
config = Config(HeuristicType.MAX_OUTPUT)

s = config.get_search(at_contrast)

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
          [at[i].get_leaf_value(n,0) for i, n in enumerate(sol_nodes[0:2])],
          [at[i].get_leaf_value(n,0) for i, n in enumerate(sol_nodes[2:4])])
### </PART>


print("\n---------------\n")
### <PART robustness0>
# Checking robustness
# We change the `base_score` of the ensemble so that we can have negative
# outputs, which is necessary for robustness checking (we want classes to
# flip!)
at.set_base_score(0,-44)

# Generate all possible output configurations for this `at`
config = Config(HeuristicType.MAX_OUTPUT)
s = config.get_search(at)

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
at = AddTree(1)
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
config = Config(HeuristicType.MAX_OUTPUT)
s = config.get_search(at)

s.steps(100)
print("\nWithout one-hot constraint")
print("{:<3} {:<10} {}".format("i", "output", "box"))
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(f"{i:<3} {sol.output:<10} {sol.box()}")
#print("number of rejected states due to constraint:", s.num_rejected_states)
### </part>

### <part onehot1>

# With constraint:
config = Config(HeuristicType.MAX_OUTPUT)
s = config.get_search(at)
#s.add_onehot_constraint([0, 1]) # TODO add again
s.steps(100)
print("\nWith one-hot constraint")
print("{:<3} {:<10} {}".format("i", "output", "box"))
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(f"{i:<3} {sol.output:<10} {sol.box()}")
#print("number of rejected states due to constraint:", s.num_rejected_states)
### </part>
