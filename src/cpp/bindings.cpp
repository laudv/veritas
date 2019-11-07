#include <pybind11/pybind11.h>
#include "domain.h"
#include "tree.h"

namespace py = pybind11;
using namespace treeck;

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
        .def("test", &LtSplit::test);

    py::class_<NodeRef>(m, "Node")
        .def("is_root", &NodeRef::is_root)
        .def("is_leaf", &NodeRef::is_leaf)
        .def("is_internal", &NodeRef::is_internal)
        .def("id", &NodeRef::id)
        .def("left", &NodeRef::left)
        .def("right", &NodeRef::right)
        .def("parent", &NodeRef::parent)
        .def("tree_size", &NodeRef::tree_size)
        .def("depth", &NodeRef::depth)
        .def("get_split", &NodeRef::get_split)
        .def("set_leaf_value", &NodeRef::set_leaf_value)
        .def("split", &NodeRef::split);
}
