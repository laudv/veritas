#include <memory>
#include <string>
#include <sstream>
#include <memory>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "domain.h"
#include "tree.h"
#include "searchspace.h"
#include "prune.h"

namespace py = pybind11;
using namespace treeck;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

using TreeD = Tree<double>;
using NodeRefD = typename TreeD::MRef;

PYBIND11_MODULE(pytreeck, m) {
    m.doc() = "Tree-CK: verification of ensembles of trees";

    py::class_<RealDomain>(m, "RealDomain")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("lo", &RealDomain::lo)
        .def_readwrite("hi", &RealDomain::hi)
        .def("contains", &RealDomain::contains)
        .def("overlaps", &RealDomain::overlaps)
        .def("is_everything", &RealDomain::is_everything)
        .def("split", &RealDomain::split)
        .def("__repr__", [](RealDomain& d) { return tostr(d); })
        .def(py::pickle(
            [](const RealDomain& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return RealDomain(t[0].cast<double>(), t[1].cast<double>());
            }));

    /* Avoid invalid pointers to Tree's by storing indexes rather than pointers */
    struct TreeRef {
        AddTree *at;
        size_t i;
        TreeD& get() { return at->operator[](i); }
        const TreeD& get() const { return at->operator[](i); }
    };

    py::class_<TreeRef>(m, "Tree")
        .def("root", [](const TreeRef& r) { return r.get().root().id(); })
        .def("index", [](const TreeRef& r) { return r.i; })
        .def("num_nodes", [](const TreeRef& r) { return r.get().num_nodes(); })
        .def("is_root", [](const TreeRef& r, NodeId n) { return r.get()[n].is_root(); })
        .def("is_leaf", [](const TreeRef& r, NodeId n) { return r.get()[n].is_leaf(); })
        .def("is_internal", [](const TreeRef& r, NodeId n) { return r.get()[n].is_internal(); })
        .def("left", [](const TreeRef& r, NodeId n) { return r.get()[n].left().id(); })
        .def("right", [](const TreeRef& r, NodeId n) { return r.get()[n].right().id(); })
        .def("parent", [](const TreeRef& r, NodeId n) { return r.get()[n].parent().id(); })
        .def("tree_size", [](const TreeRef& r, NodeId n) { return r.get()[n].tree_size(); })
        .def("depth", [](const TreeRef& r, NodeId n) { return r.get()[n].depth(); })
        .def("get_leaf_value", [](const TreeRef& r, NodeId n) { return r.get()[n].leaf_value(); })
        .def("get_split", [](const TreeRef& r, NodeId n) -> std::tuple<FeatId, double> {
                auto split = std::get<LtSplit>(r.get()[n].get_split());
                return {split.feat_id, split.split_value}; })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, double v) { r.get()[n].set_leaf_value(v); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, double sv) { r.get()[n].split(LtSplit(fid, sv)); })
        .def("skip_branch", [](TreeRef& r, NodeId n) { r.get()[n].skip_branch(); })
        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); });

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        .def_readwrite("base_score", &AddTree::base_score)
        .def("__len__", &AddTree::size)
        .def("num_nodes", &AddTree::num_nodes)
        .def("add_tree", [](AddTree& at) -> TreeRef { return TreeRef{&at, at.add_tree(TreeD())}; } )
        .def("__getitem__", [](AddTree& at, size_t i) -> TreeRef { return TreeRef{&at, i}; })
        .def("use_count", [](const std::shared_ptr<AddTree>& at) { return at.use_count(); })
        .def("get_splits", &AddTree::get_splits)
        .def("to_json", &AddTree::to_json)
        .def("from_json", AddTree::from_json)
        .def("__str__", [](const AddTree& at) { return tostr(at); });

    py::class_<SearchSpace>(m, "SearchSpace")
        .def(py::init<std::shared_ptr<AddTree>>())
        .def(py::init<std::shared_ptr<AddTree>, const Domains::vec_t&>())
        .def("split", [](SearchSpace& sp, size_t nleafs) {
            sp.split(UnreachableNodesMeasure{}, NumDomTreeLeafsStopCond{nleafs});
        })
        .def("num_features", &SearchSpace::num_features)
        .def("scores", &SearchSpace::scores)
        .def("leafs", &SearchSpace::leafs)
        .def("get_domains", [](SearchSpace& sp, NodeId leaf_id) {
            Domains doms;
            sp.get_domains(leaf_id, doms);
            return doms.vec();
        })
        .def("get_pruned_addtree", [](const SearchSpace& sp, NodeId node_id) {
            Domains doms;
            sp.get_domains(node_id, doms);
            AddTree new_at = prune(sp.addtree(), doms);
            return new_at;
        });

} /* PYBIND11_MODULE */


