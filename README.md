[![PyPi Version](https://img.shields.io/pypi/v/dtai-veritas)](https://pypi.org/project/dtai-veritas/)

> ℹ️ We're working on an improved version of Veritas. If you are interested in trying out Veritas and want the latest and greatest, then pull and manually install the [finiteprec](https://github.com/laudv/veritas/tree/finiteprec) branch.

# Versatile Verification of Tree Ensembles with VERITAS

[View API documentation](https://alexandersch12.github.io/veritas/) | [Veritas in action blog post](https://dtai.cs.kuleuven.be/sports/blog/versatile-verification-of-soccer-analytics-models/)

## Installation

- Clone this repository: `git clone https://github.com/laudv/veritas.git`
- Change directory: `cd veritas`
- Initialize the [pybind11](https://pybind11.readthedocs.io) submodule `git submodule init` and `git submodule update`
- If you use environments: activate a (new) Python3 environment (e.g. using `venv`: `python3 -m venv venv_name && source venv_name/bin/activate`)
- run `pip install .` in the root directory of Veritas

Veritas should work on Linux (GCC), Mac (LLVM), and Windows (MSVC). If you encounter issues, feel free to contact me or open an issue.

To pull the latest updates from Github, simply `git pull` the changes and reinstall using `pip`: `pip install --force-reinstall .`.

## Constructing an Additive Tree Ensemble or `AddTree`

You can convert an existing ensemble using the `get_addtree` function for XGBoost, LightGBM and scikit-learn.

Here's an example of a model trained by XGBoost that has been converted to Veritas' own tree ensemble representation.

```python
from veritas import *
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X,Y) = make_moons(100)

clf = RandomForestClassifier(
        max_depth=4,
        random_state=0,
        n_estimators=3)

trained_model = clf.fit(X, Y)

# Convert the RandomForestClassifier model to a Veritas tree ensemble
addtree = get_addtree(trained_model)

print(f"{addtree}\n")

# Print all trees in the ensemble
for tree in addtree:
    print(tree)
```

The output is an AddTree consisting of 3 trees, as was defined in the XGBClassifier.

```
SKLEARN: classifier with 1 classes
AddTree with 3 trees and base_scores [0]

Node(id=0, split=[F0 < -0.0160258], sz=9, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[1])
├─ Node(id=2, split=[F1 < 0.522767], sz=7, left=3, right=4)
│  ├─ Node(id=3, split=[F0 < 0.839224], sz=5, left=5, right=6)
│  │  ├─ Leaf(id=5, sz=1, value=[0])
│  │  ├─ Node(id=6, split=[F1 < 0.373695], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.666667])
│  └─ Leaf(id=4, sz=1, value=[1])

Node(id=0, split=[F1 < 0.126305], sz=15, left=1, right=2)
├─ Node(id=1, split=[F1 < -0.0227674], sz=7, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[0])
│  ├─ Node(id=4, split=[F1 < 0.00464122], sz=5, left=5, right=6)
│  │  ├─ Leaf(id=5, sz=1, value=[1])
│  │  ├─ Node(id=6, split=[F1 < 0.0366763], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.4])
├─ Node(id=2, split=[F0 < 1.48667], sz=7, left=9, right=10)
│  ├─ Node(id=9, split=[F1 < 0.522767], sz=5, left=11, right=12)
│  │  ├─ Node(id=11, split=[F1 < 0.434907], sz=3, left=13, right=14)
│  │  │  ├─ Leaf(id=13, sz=1, value=[0.823529])
│  │  │  └─ Leaf(id=14, sz=1, value=[0.25])
│  │  └─ Leaf(id=12, sz=1, value=[1])
│  └─ Leaf(id=10, sz=1, value=[0])

Node(id=0, split=[F1 < -0.0490553], sz=11, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[0])
├─ Node(id=2, split=[F1 < 0.463324], sz=9, left=3, right=4)
│  ├─ Node(id=3, split=[F1 < 0.156384], sz=7, left=5, right=6)
│  │  ├─ Node(id=5, split=[F1 < 0.126305], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0.8])
│  │  │  └─ Leaf(id=8, sz=1, value=[1])
│  │  ├─ Node(id=6, split=[F1 < 0.373695], sz=3, left=9, right=10)
│  │  │  ├─ Leaf(id=9, sz=1, value=[0.294118])
│  │  │  └─ Leaf(id=10, sz=1, value=[0.625])
│  └─ Leaf(id=4, sz=1, value=[1])

```

## Queries

Starting from this AddTree, we can perform several queries.

```
AddTree with 2 trees and base_scores [0]

Node(id=0, split=[F0 < 2], sz=7, left=1, right=2)
├─ Node(id=1, split=[F0 < 1], sz=3, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[3])
│  └─ Leaf(id=4, sz=1, value=[4])
├─ Node(id=2, split=[F0 < 3], sz=3, left=5, right=6)
│  ├─ Leaf(id=5, sz=1, value=[5])
│  └─ Leaf(id=6, sz=1, value=[6])

Node(id=0, split=[F0 < 3], sz=9, left=1, right=2)
├─ Node(id=1, split=[F1 < 5], sz=3, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[30])
│  └─ Leaf(id=4, sz=1, value=[40])
├─ Node(id=2, split=[F1 < 0], sz=5, left=5, right=6)
│  ├─ Leaf(id=5, sz=1, value=[50])
│  ├─ Node(id=6, split=[F2 < 0.5], sz=3, left=7, right=8)
│  │  ├─ Leaf(id=7, sz=1, value=[70])
│  │  └─ Leaf(id=8, sz=1, value=[80])

```

### Finding the Global Maximum of the Ensemble

We can use Veritas to find the feature values for which the model's output is maximal as follows.

```python
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
```

This outputs:

```
Global maximum
- current best solution: 86.0 -> optimal solution
- feature value ranges {0: Interval(>=3), 1: Interval(>=0), 2: Interval(>=0.5)}
  which lead to leaf nodes [6, 8] with leaf values [6.0, 80.0]
```

The solutions generated by `Search` are accessible using `get_solution`. The solutions are sorted descendingly: the best solution is at index 0, the worst solution is at index `s.num_solutions()-1`.

A best solution at index 0 is optimal when `s.is_optimal()` returns true. To know when the solution was generated, `sol.time` contains the number of seconds since the construction of the `Search` object.

The `sol.box()` method returns the value intervals of the features for which the output of the ensemble is unchanged. That is, for each possible assignment within the intervals, the trees always evaluate to the same leaf node (`s.get_solution_nodes`), and thus to the same output value. If a feature is missing from the box, it means that its value does not make a difference.

### Constrained Minimization

In this example, we constrain the first feature value to be between 3 and 5.
Because this is a very simple constraint, we can simply prune the search space before we start the search.

Although the constraint is simple, it is very useful. The exact same pruning strategy is used for l-infinity robustenss checking.

```python
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
```

The output is:

```
Minimum with feature0 in [3, 5]
- current best solution: -56.0 -> optimal solution
- feature value ranges {0: Interval(3,5), 1: Interval(<0)}
  which lead to leaf nodes [6, 5] with leaf values [6.0, 50.0]
```

We minimize by maximizing the negated ensemble, i.e., the ensemble where all leaf values are negated.

The pruning simply removes all leaf nodes with boxes that do not overlap with `prune_box` from the search.

### Contrasting Two Instances

In this example, we want to know what the maximum difference between the outputs of two instances can be when only the third feature is different, and first and second feature values are the same.

We achieve this by renaming the feature IDs in one of the trees using a feature map or `FeatMap` object.

```python
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
```

Output:
```
feat_id used for feature3 for instances: 2 5

Node(id=0, split=[F0 < 3], sz=9, left=1, right=2)
├─ Node(id=1, split=[F1 < 5], sz=3, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[30])
│  └─ Leaf(id=4, sz=1, value=[40])
├─ Node(id=2, split=[F1 < 0], sz=5, left=5, right=6)
│  ├─ Leaf(id=5, sz=1, value=[50])
│  ├─ Node(id=6, split=[F2 < 0.5], sz=3, left=7, right=8)
│  │  ├─ Leaf(id=7, sz=1, value=[70])
│  │  └─ Leaf(id=8, sz=1, value=[80])

Node(id=0, split=[F0 < 3], sz=9, left=1, right=2)
├─ Node(id=1, split=[F1 < 5], sz=3, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[-30])
│  └─ Leaf(id=4, sz=1, value=[-40])
├─ Node(id=2, split=[F1 < 0], sz=5, left=5, right=6)
│  ├─ Leaf(id=5, sz=1, value=[-50])
│  ├─ Node(id=6, split=[F5 < 0.5], sz=3, left=7, right=8)
│  │  ├─ Leaf(id=7, sz=1, value=[-70])
│  │  └─ Leaf(id=8, sz=1, value=[-80])

```

There are two differences between tree 1 and tree 3:

- the leaf values are negated (`concat_negated`)
- internal node 6 uses feature ID 2 in tree 1 and feature ID 5 in tree 3

The other feature IDs are the same. This has the effect of allowing the first two trees (corresponding to the first instance) to take on different values for feature 3 than the last two trees (corresponding to the second instance).

The renaming of the feature IDs is fascilitated by the `FeatMap` object.

```python
print(feat_map)
```
```
FeatMap {
    [0] `feature1` -> 0 (instance 0)
    [1] `feature2` -> 1 (instance 0)
    [2] `feature3` -> 2 (instance 0)
    [3] `feature1` -> 0 (instance 1)
    [4] `feature2` -> 1 (instance 1)
    [5] `feature3` -> 5 (instance 1)
}
```

The above gives all IDs used by the two instances. `FeatMap::share_all_features_between_instances` can be used share all feature values between the two intances. By default, each ID is unique.
Use `FeatMap::use_same_id_for` to share the same ID for two features, either between two instances, or for the same instance.
Use `FeatMap::transform` to apply the changes to an `AddTree`.

We can find the maximum difference between the outputs of the first and the second instance as follows:

```python
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
```

Output:
```
Maximum difference between instance0 and instance1
- current best solution: 10.0 -> optimal solution
- feature value ranges {0: Interval(>=3), 1: Interval(>=0), 2: Interval(>=0.5), 5: Interval(<0.5)}
  which lead to leaf nodes [6, 8, 6, 7] with leaf values [6.0, 80.0] [6.0, 70.0]
```

The maximum output difference in this case is 10. The only possible variation is between leaf nodes 7 or 8 in the second tree.

Use `Search::step_for(duration_in_seconds, num_steps)` to let the search run for the given duration. Per `num_steps` steps, a snapshot is added to `Search::snapshots`. This can be used to track the following stats:

- time (`time`)
- number of steps executed so far (`num_steps`)
- number of solutions so far (`num_solutions`)
- number of search states expanded so far (`num_states`)
- best epsilon value (`eps`)
- the best bounds so far (`bounds`), a tuple containing lower bound, A\* upper bound, and ARA\* upper bound

### Checking Robustness

Before we check the robustness of a particular example, we'll first use Veritas to enumerate all possible output configurations of the additive tree ensemble. To do this, we simply run the search until `Search::steps` returns false, indicating that all search states have been visited.

```python
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
```

```
i   output     box
0   42.0       {0: Interval(>=3), 1: Interval(>=0), 2: Interval(>=0.5)}
```

The boxes above partition the input space. Remember that when a feature is not present in a box, it does not have an effect given the other feature values and can take on any value.

We will pick an example from box 6 with output -9:

```python
example = [2, 4, 2]
print("output for example", example, "is", at.eval(example)[0])
```

Output:
```
output for example [2, 4, 2] is [-9.]
```

We now try to find the distance to the closest adversarial example for which the output of the model is positive. We use `VeritasRobustnessSearch` for this. The arguments are:

- model to minimize or None
- model to maximize or None (use both for targeted attacks)
- the example
- the initial delta value used by the binary search

```python
from veritas import VeritasRobustnessSearch
rob = VeritasRobustnessSearch(None, at, example, start_delta=5.0)
delta, delta_lo, delta_up = rob.search()

print("adversarial examples:", rob.generated_examples,
        "with outputs", at.eval(np.array(rob.generated_examples)))
```

Output:
```
[0 0.0s]:   SAT for delta 5.00000 -> 0.50000 [0.00000, 1.00000] (!) ex.w/ delta 1.0000
[1 0.0s]: UNSAT for delta 0.50000 -> 0.75000 [0.50000, 1.00000]
[2 0.0s]: UNSAT for delta 0.75000 -> 0.87500 [0.75000, 1.00000]
[3 0.0s]: UNSAT for delta 0.87500 -> 0.93750 [0.87500, 1.00000]
[4 0.0s]: UNSAT for delta 0.93750 -> 0.96875 [0.93750, 1.00000]
[5 0.0s]: UNSAT for delta 0.96875 -> 0.98438 [0.96875, 1.00000]
[6 0.0s]: UNSAT for delta 0.98438 -> 0.99219 [0.98438, 1.00000]
[7 0.0s]: UNSAT for delta 0.99219 -> 0.99609 [0.99219, 1.00000]
[8 0.0s]: UNSAT for delta 0.99609 -> 0.99805 [0.99609, 1.00000]
[9 0.0s]: UNSAT for delta 0.99805 -> 0.99902 [0.99805, 1.00000]
adversarial examples: [[3.0, 4, 2]] with outputs [[42.]]
```

We can verify this result using the MILP approach (Kantchelian et al.'16):

```python
from veritas.kantchelian import KantchelianAttack

kan = KantchelianAttack(at, target_output=True, example=example, silent=True)
kan.optimize()
adv_example, adv_output = kan.solution()[:2]
print("Kantchelian adversarial example", adv_example, "with output", adv_output)
```

Output:
```
Kantchelian adversarial example [3.0, 4, 2] with output 42.0
```

MILP indeed finds the same solution.

### One-hot constraint

We can tell Veritas that some of the features are the results of a one-hot encoded categorical feature using `Search::add_onehot_constraint`. This ensures that exactly one of the features is true at all times.

For this constructed example with only two one-hot encoded features, the total number of solutions is four, but two of them are invalid:

```python
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
```

Output:

```
Node(id=0, split=[F0 < 0.5], sz=3, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[-100])
└─ Leaf(id=2, sz=1, value=[100])

Node(id=0, split=[F1 < 0.5], sz=3, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[-100])
└─ Leaf(id=2, sz=1, value=[100])


Without one-hot constraint
i   output     box
0   200.0      {0: Interval(>=0.5), 1: Interval(>=0.5)}
```

When we inform Veritas that exactly one of the two features must be true:

```python

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
```

Output:

```

With one-hot constraint
i   output     box
0   200.0      {0: Interval(>=0.5), 1: Interval(>=0.5)}
```

# Citate

[View API documentation](https://alexandersch12.github.io/veritas/) | [Veritas in action blog post](https://dtai.cs.kuleuven.be/sports/blog/versatile-verification-of-soccer-analytics-models/)

**Veritas** is a versatile verification tool for tree ensembles. You can use
Veritas to generate _adversarial examples_, check _robustness_, find _dominant
attributes_ or simply ask _domain specific questions_ about your model.

Veritas uses its own tree representation and does not assume a specific model format (like XGBoost's JSON dump).
This makes it easy to use with many tree/ensemble learners. A translation function is included for XGBoost ensembles.

For more information, refer to the paper:

> Versatile Verification of Tree Ensembles.
> Laurens Devos, Wannes Meert, and Jesse Davis.
> ICML 2021
> http://proceedings.mlr.press/v139/devos21a.html
