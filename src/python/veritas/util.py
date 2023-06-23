## \file util.py
# \defgroup python Python classes and functions
# \brief An overview of all class and functions defined in Python.

from veritas import Solution

## \ingroup python
# \brief Generate a concrete instance.
#
# Generate a instance from a box (either a list of domains, one for each
# feature, a list of (feat_id, dom) pairs, or a dict) or from a
# GraphSearch::solution.
#
# Take `example = np.zeros(num_attributes)` if you just want an example.
#
# \param featmap Mapping of feature ids, see FeatMap
def get_closest_example(solution_or_box, example, eps, featmap=None):
    num_attributes = len(example)

    if featmap is None:
        featmap = {i: [i] for i in range(num_attributes)}
    else:
        featmap = featmap.get_indices_map()

    closest = example.copy()

    if isinstance(solution_or_box, Solution):
        box = solution_or_box.box()
    elif isinstance(solution_or_box, list):
        if isinstance(solution_or_box[0], tuple):
            box = {x[0]: x[1] for x in solution_or_box}
        else:
            box = {i: x for i, x in enumerate(solution_or_box)}
    elif isinstance(solution_or_box, dict):
        box = solution_or_box
    else:
        raise ValueError("invalid first argument")

    for index, dom in box.items():
        for feat_id in featmap[index]:
            x = example[feat_id]
            if dom.lo <= x and x < dom.hi:
                continue # keep the value x

            dist_lo = abs(dom.lo - x)
            dist_hi = abs(x - dom.hi)
            if dist_lo > dist_hi:
                closest[feat_id] = dom.hi - eps
            else:
                closest[feat_id] = dom.lo

            #print(f"dom {feat_id}:", dom, x, "->", closest[feat_id])

    return closest


# PYTHON BINDINGS DOCS

## \ingroup python
# \class Domain
# \brief Bindings to C++ veritas::Domain class.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<Domain>
# \until ; // Domain

## \ingroup python
# \class DomainPair
# \brief Bindings to C++ veritas::DomainPair struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<DomainPair>
# \until ; // DomainPair

## \ingroup python
# \class LtSplit
# \brief Bindings to C++ veritas::LtSplit struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<LtSplit>
# \until ; // LtSplit

## \ingroup python
# \class Tree
# \brief Bindings to C++ veritas::Tree struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<TreeRef>
# \until ; // TreeRef

## \ingroup python
# \class AddTree
# \brief Bindings to C++ veritas::AddTree struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<AddTree
# \until ; // AddTree

## \ingroup python
# \class FeatMap
# \brief Bindings to C++ veritas::FeatMap struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<FeatMap>
# \until ; // FeatMap

## \ingroup python
# \class GraphOutputSearch
# \brief Bindings to C++ veritas::GraphOutputSearch class.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<GraphOutputSearch>
# \until ; // GraphOutputSearch

## \ingroup python
# \class GraphRobustnessSearch
# \brief Bindings to C++ veritas::GraphRobustnessSearch struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<GraphRobustnessSearch>
# \until ; // GraphRobustnessSearch

## \ingroup python
# \class Snapshot
# \brief Bindings to C++ veritas::Snapshot struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<Snapshot>
# \until ; // Snapshot

## \ingroup python
# \class Solution
# \brief Bindings to C++ veritas::Solution struct.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<Solution>
# \until ; // Solution

## \ingroup python
# \class Search
# \brief Bindings to C++ veritas::Search class.
#
# In `bindings.cpp`:
# \dontinclude[lineno] bindings.cpp
# \skipline py::class_<VSearch>
# \until ; // VSearch
