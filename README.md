# Versatile Verification of Tree Ensembles with VERITAS

See the paper for more information: https://arxiv.org/abs/2010.13880

## Dependencies

* c++
* cmake
* pybind11 (included)
* cereal (included)
* python3
* z3 (optional)

You can use Ubuntu's and Fedora's Z3 package. If you prefer your local
installation, set the CMAKE variable `Z3_INSTALL` to the root directory of your
Z3 installation.


## Installation

* Clone this repository.
* Initialize the submodules `git submodule init` and `git submodule update`.
* Activate a (new) Python3 environment.
* run `pip install .` in the root directory of Veritas

If you want to use Z3, modify CMake variables `USE_Z3` and optionally
`Z3_INSTALL`. Z3 is disabled by default.


## Example

```python
from veritas import *

# Manually create a two-tree ensemble
at = AddTree()
t = at.add_tree();
t.split(t.root(), 0, 2)
t.split( t.left(t.root()), 0, 1)
t.split(t.right(t.root()), 0, 3)
t.set_leaf_value( t.left( t.left(t.root())), 0.1)
t.set_leaf_value(t.right( t.left(t.root())), 0.2)
t.set_leaf_value( t.left(t.right(t.root())), 0.3)
t.set_leaf_value(t.right(t.right(t.root())), 0.4)

t = at.add_tree();
t.split(t.root(), 0, 3)
t.split( t.left(t.root()), 1, 1.2)
t.split(t.right(t.root()), 1, 3.3)
t.split(t.right(t.right(t.root())), 2)
t.set_leaf_value( t.left( t.left(t.root())), 0.1)
t.set_leaf_value(t.right( t.left(t.root())), 0.2)
t.set_leaf_value( t.left(t.right(t.root())), 0.3)
t.set_leaf_value( t.left(t.right(t.right(t.root()))), 0.5)
t.set_leaf_value(t.right(t.right(t.right(t.root()))), 0.6)

# Print the simple additive ensemble
print(at)

# Initialize the optimizer maximizing F(x_2) - F(x_1), where x_1 and x_2 share
# a feature value for attribute 1.
opt = Optimizer(minimize=at, maximize=at, matches={1}, match_is_reuse=True);

print("num_vertices_in_set g0 0", opt.g0.num_vertices_in_set(0), "should be", 4)
print("num_vertices_in_set g0 1", opt.g0.num_vertices_in_set(1), "should be", 5)
print("num_vertices_in_set g1 0", opt.g1.num_vertices_in_set(0), "should be", 4)
print("num_vertices_in_set g1 1", opt.g1.num_vertices_in_set(1), "should be", 5)
print()

# Use the Z3 SMT solver to prune the ensemble: enforce that
#  - x_10 < 0.0, 
#  - x_11 > 0.0 (x_11 == x_21),
#  - x_20 > 1.0
opt.prune_smt("""
(assert (< {instance1_attribute0} 0.0))
(assert (> {instance1_attribute1} 1.2))
(assert (> {instance2_attribute0} 1.0))""",
    var_prefix0="instance1_attribute",
    var_prefix1="instance2_attribute")

print("num_vertices_in_set g0 0", opt.g0.num_vertices_in_set(0), "should be", 1)
print("num_vertices_in_set g0 1", opt.g0.num_vertices_in_set(1), "should be", 1)
print("num_vertices_in_set g1 0", opt.g1.num_vertices_in_set(0), "should be", 3)
print("num_vertices_in_set g1 1", opt.g1.num_vertices_in_set(1), "should be", 4)
print()

# Use the Veritas algorithm to find all solutions
notdone = opt.steps(100)
print("number of solutions is", opt.num_solutions())
print("still more solutions" if notdone else "we found all solutions")

print("listing all solutions:")
for sol in opt.solutions():
    print(sol)
print()

# Use Chen et al. (2019)'s Merge algorithm:
g = opt.g1
g.add_with_negated_leaf_values(opt.g0) # Merge's way of dealing with two models
print("merge bounds:", g.basic_bound())
print("merge number of independent sets:", g.num_independent_sets(), "should be", 4)
g.merge(2)
print("merge bounds:", g.basic_bound())
print("merge number of independent sets:", g.num_independent_sets(), "should be", 2)
g.merge(2)
print("merge bounds:", g.basic_bound())
print("merge number of independent sets:", g.num_independent_sets(), "should be", 1)
```

You can convert XGBoost models to `AddTree` objects using the
`veritas.xgb.addtrees_from_multiclass_xgb_model` function.
