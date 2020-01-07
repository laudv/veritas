#include <memory>
#include <string>
#include <sstream>
#include <memory>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

#include "domain.h"
#include "tree.h"
#include "subspaces.h"

namespace py = pybind11;
using namespace treeck;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

static
py::tuple
encode_split(const Split& split)
{
    return visit_split(
        [](const LtSplit& s) -> py::tuple {
            return py::make_tuple("lt", s.feat_id, s.split_value);
        },
        [](const BoolSplit& s) -> py::tuple {
            return py::make_tuple("bool", s.feat_id);
        },
        split);
}


using TreeD = Tree<FloatT>;
using NodeRefD = TreeD::MRef;
using DomTreeT = Subspaces::DomTreeT;


PYBIND11_MODULE(pytreeck, m) {
    m.doc() = "Tree-CK: verification of ensembles of trees";

    py::class_<RealDomain>(m, "RealDomain")
        .def(py::init<>())
        .def(py::init<FloatT, FloatT>())
        .def_readwrite("lo", &RealDomain::lo)
        .def_readwrite("hi", &RealDomain::hi)
        .def("contains", &RealDomain::contains)
        .def("overlaps", &RealDomain::overlaps)
        .def("is_everything", &RealDomain::is_everything)
        .def("split", &RealDomain::split)
        .def("__repr__", [](const RealDomain& d) { return tostr(d); })
        .def(py::pickle(
            [](const RealDomain& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return RealDomain(t[0].cast<FloatT>(), t[1].cast<FloatT>());
            }));

    py::class_<BoolDomain>(m, "BoolDomain")
        .def(py::init<>())
        .def(py::init<bool>())
        .def_readonly("_value", &BoolDomain::value_)
        .def("is_everything", &BoolDomain::is_everything)
        .def("is_true", &BoolDomain::is_true)
        .def("is_false", &BoolDomain::is_false)
        .def("split", &BoolDomain::split)
        .def("__repr__", [](BoolDomain& d) { return tostr(d); })
        .def(py::pickle(
            [](const BoolDomain& d) { return py::make_tuple(d.value_); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 1) throw std::runtime_error("invalid pickle state");
                BoolDomain dom;
                dom.value_ = t[0].cast<int>();
                return dom;
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
        .def("get_split", [](const TreeRef& r, NodeId n) { return encode_split(r.get()[n].get_split()); })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, FloatT v) { r.get()[n].set_leaf_value(v); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, FloatT sv) { r.get()[n].split(LtSplit(fid, sv)); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid) { r.get()[n].split(BoolSplit(fid)); })
        .def("skip_branch", [](TreeRef& r, NodeId n) { r.get()[n].skip_branch(); })
        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); });

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        .def_readwrite("base_score", &AddTree::base_score)
        .def("__len__", &AddTree::size)
        .def("num_nodes", &AddTree::num_nodes)
        .def("num_leafs", &AddTree::num_leafs)
        .def("add_tree", [](AddTree& at) -> TreeRef { return TreeRef{&at, at.add_tree(TreeD())}; } )
        .def("__getitem__", [](AddTree& at, size_t i) -> TreeRef { return TreeRef{&at, i}; })
        .def("use_count", [](const std::shared_ptr<AddTree>& at) { return at.use_count(); })
        .def("get_splits", &AddTree::get_splits)
        .def("to_json", &AddTree::to_json)
        .def("from_json", AddTree::from_json)
        .def("__str__", [](const AddTree& at) { return tostr(at); })
        .def(py::pickle(
            [](const AddTree& at) { // __getstate__
                return at.to_json();
            },
            [](const std::string& json) { // __setstate__
                return AddTree::from_json(json);
            }));

    py::class_<Subspaces>(m, "Subspaces")
        .def(py::init<std::shared_ptr<AddTree>, Subspaces::DomainsT>())
        .def("domtree", &Subspaces::domtree)
        .def("addtree", &Subspaces::addtree)
        .def("get_root_domain", &Subspaces::get_root_domain)
        .def("get_domains", [](const Subspaces& st, NodeId n) {
            Subspaces::DomainsT domains;
            st.get_domains(n, domains);
            return domains;
        })
        .def("get_subspace", &Subspaces::get_subspace)
        .def("split_leaf", &Subspaces::split_domtree_leaf)
        .def("to_json", &Subspaces::to_json)
        .def("from_json", &Subspaces::from_json)
        .def("split", &Subspaces::split_domtree_leaf)
        .def("split", [](Subspaces& st, Subspace& subspace) {
            st.split(std::move(subspace));
            // C++ standard specifies `leaf` is in valid but unspecified state afterwards
            // Python users really shouldn't reuse `leaf` afterwards
        });

    py::class_<Subspace>(m, "Subspace")
        .def_readonly("split_score", &Subspace::split_score)
        .def_readonly("split_balance", &Subspace::split_balance)
        .def("domtree_node_id", &Subspace::domtree_node_id)
        .def("get_domains", &Subspace::get_domains)
        .def("get_domain", &Subspace::get_domain)
        .def("num_unreachable", &Subspace::num_unreachable)
        .def("is_reachable", &Subspace::is_reachable)
        .def("mark_unreachable", &Subspace::mark_unreachable)
        .def("find_best_domtree_split", &Subspace::find_best_domtree_split)
        .def("has_best_split", &Subspace::has_best_split)
        .def("get_best_split", [](const Subspace& ss) { return encode_split(ss.get_best_split()); })
        .def("get_tree_bounds", &Subspace::get_tree_bounds)
        .def("merge", &Subspace::merge)
        .def("to_json", &Subspace::to_json)
        .def("from_json", &Subspace::from_json)
        .def(py::pickle(
            [](const Subspace& p) { // __getstate__
                return p.to_json();
            },
            [](const std::string& json) { // __setstate__
                return Subspace::from_json(json);
            }));

    py::class_<DomTreeT>(m, "DomTree")
        .def("root", [](const DomTreeT& t) { return t.root().id(); })
        .def("num_nodes", [](const DomTreeT& t) { return t.num_nodes(); })
        .def("is_root", [](const DomTreeT& t, NodeId n) { return t[n].is_root(); })
        .def("is_leaf", [](const DomTreeT& t, NodeId n) { return t[n].is_leaf(); })
        .def("is_internal", [](const DomTreeT& t, NodeId n) { return t[n].is_internal(); })
        .def("left", [](const DomTreeT& t, NodeId n) { return t[n].left().id(); })
        .def("right", [](const DomTreeT& t, NodeId n) { return t[n].right().id(); })
        .def("parent", [](const DomTreeT& t, NodeId n) { return t[n].parent().id(); })
        .def("__str__", [](const DomTreeT& at) { return tostr(at); })
        .def("get_split", [](const DomTreeT& t, NodeId n) { return encode_split(t[n].get_split()); });

} /* PYBIND11_MODULE */
