#include <memory>
#include <string>
#include <sstream>
#include <memory>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "domain.h"
#include "tree.h"
//#include "opaque.h"

namespace py = pybind11;
using namespace treeck;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

PYBIND11_MODULE(treeck, m) {
    m.doc() = "Tree-CK: verification of ensembles of trees";

    py::class_<RealDomain>(m, "RealDomain")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readonly("lo", &RealDomain::lo)
        .def_readonly("hi", &RealDomain::hi)
        .def("contains", &RealDomain::contains)
        .def("overlaps", &RealDomain::overlaps)
        .def("split", &RealDomain::split);

    py::class_<LtSplit>(m, "LtSplit")
        .def(py::init<FeatId, LtSplit::ValueT>())
        .def_readonly("feat_id", &LtSplit::feat_id)
        .def_readonly("split_value", &LtSplit::split_value)
        .def("test", &LtSplit::test)
        .def("__repr__", [](LtSplit& s) { return tostr(s); });

    py::class_<NodeRef>(m, "Node")
        .def("is_root", &NodeRef::is_root)
        .def("is_leaf", &NodeRef::is_leaf)
        .def("is_internal", &NodeRef::is_internal)
        .def("id", &NodeRef::id)
        .def("left", &NodeRef::left, py::keep_alive<0, 1>())   // keep_alive<Nurse, Patient> = <Return, This>, 
        .def("right", &NodeRef::right, py::keep_alive<0, 1>()) // Patient kept alive until Nurse dropped
        .def("parent", &NodeRef::parent, py::keep_alive<0, 1>())
        .def("tree_size", &NodeRef::tree_size)
        .def("depth", &NodeRef::depth)
        .def("get_split", &NodeRef::get_split)
        .def("leaf_value", &NodeRef::leaf_value)
        .def("set_leaf_value", &NodeRef::set_leaf_value)
        .def("split", [](NodeRef& n, LtSplit s) { n.split(s); })
        .def("__repr__", [](NodeRef& n) { return tostr(n); });
    
    py::class_<Tree/*, std::unique_ptr<Tree, py::nodelete>*/>(m, "Tree")
        .def(py::init<>())
        .def("root", &Tree::root, py::keep_alive<0, 1>())
        .def("__getitem__", &Tree::operator[], py::keep_alive<0, 1>())
        .def("__setitem__", [](Tree& tree, NodeId id, double leaf_value) {
            NodeRef node = tree[id];
            if (node.is_internal()) throw std::runtime_error("set leaf value of internal");
            node.set_leaf_value(leaf_value);
        })
        .def("num_nodes", &Tree::num_nodes)
        .def("to_json", &Tree::to_json)
        .def("id", &Tree::id)
        .def("from_json", &Tree::from_json)
        .def("__repr__", [](Tree& t) { return tostr(t); });

    /* Avoid invalid pointers to Tree's by storing indexes rather than pointers */
    struct TreeRef {
        AddTree *at;
        size_t i;
        Tree& get() { return at->operator[](i); }
    };

    py::class_<TreeRef>(m, "TreeRef")
        .def("root", [](TreeRef& r) { return r.get().root(); }, py::keep_alive<0, 1>())
        .def("__getitem__", [](TreeRef& r, NodeId id) { return r.get()[id]; }, py::keep_alive<0, 1>())
        .def("__setitem__", [](TreeRef& r, NodeId id, double leaf_value) {
            NodeRef node = r.get()[id];
            if (node.is_internal()) throw std::runtime_error("set leaf value of internal");
            node.set_leaf_value(leaf_value);
        })
        .def("num_nodes", [](TreeRef& r) { return r.get().num_nodes(); })
        .def("id", [](TreeRef& r) { return r.get().id(); })
        .def("__repr__", [](TreeRef& r) { return tostr(r.get()); });


    py::class_<AddTree>(m, "AddTree")
        .def(py::init<>())
        .def("__len__", [](AddTree& at) { return at.size(); })
        .def("new_tree", [](AddTree& at) -> TreeRef { return TreeRef{&at, at.add_tree(Tree())}; } )
        .def("__getitem__", [](AddTree& at, size_t i) -> TreeRef { return TreeRef{&at, i}; })
        .def("to_json", &AddTree::to_json)
        .def("from_json", &AddTree::from_json);
}
