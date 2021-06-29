/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>

#include "domain.hpp"
#include "new_tree.hpp"

#ifdef VERITAS_FEATURE_SMT
    #include <z3++.h>
    #include "smt.h"
#endif

namespace py = pybind11;
using namespace veritas;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

//static AddTree DUMMY_ADDTREE{};

//using TreeD = Tree<Split, FloatT>;
//using NodeRefD = TreeD::MRef;

PYBIND11_MODULE(pyveritas, m) {
    m.doc() = "Veritas: verification of tree ensembles";

    py::class_<Domain>(m, "Domain")
        .def(py::init<>())
        .def(py::init<FloatT, FloatT>())
        .def_static("from_lo", &Domain::from_lo)
        .def_static("from_hi_exclusive", &Domain::from_hi_exclusive)
        .def_static("from_hi_inclusive", &Domain::from_hi_inclusive)
        .def_static("exclusive", &Domain::exclusive)
        .def_static("inclusive", &Domain::inclusive)
        .def_readwrite("lo", &Domain::lo)
        .def_readwrite("hi", &Domain::hi)
        .def("lo_is_inf", &Domain::lo_is_inf)
        .def("hi_is_inf", &Domain::hi_is_inf)
        .def("contains", &Domain::contains)
        .def("overlaps", &Domain::overlaps)
        .def("is_everything", &Domain::is_everything)
        .def("split", &Domain::split)
        .def("__eq__", [](const Domain& s, const Domain& t) { return s == t; })
        .def("__repr__", [](const Domain& d) { return tostr(d); })
        //.def(py::pickle(
        //    [](const Domain& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
        //    [](py::tuple t) { // __setstate__
        //        if (t.size() != 2) throw std::runtime_error("invalid pickle state");
        //        return Domain(t[0].cast<FloatT>(), t[1].cast<FloatT>());
        //    }))
        ;

    py::class_<LtSplit>(m, "LtSplit")
        .def(py::init<FeatId, FloatT>())
        .def_readonly("feat_id", &LtSplit::feat_id)
        .def_readonly("split_value", &LtSplit::split_value)
        .def("test", &LtSplit::test)
        .def("__eq__", [](const LtSplit& s, const LtSplit& t) { return s == t; })
        .def("__repr__", [](const LtSplit& s) { return tostr(s); })
        //.def(py::pickle(
        //    [](const LtSplit& s) { return py::make_tuple(s.feat_id, s.split_value); }, // __getstate__
        //    [](py::tuple t) -> LtSplit { // __setstate__
        //        if (t.size() != 2) throw std::runtime_error("invalid pickle state");
        //        return { t[0].cast<FeatId>(), t[1].cast<FloatT>() };
        //    }))
        ;


    /* Avoid invalid pointers to Tree's by storing indexes rather than pointers */
    struct TreeRef {
        std::shared_ptr<AddTree> at;
        size_t i;
        Tree& get() { return at->operator[](i); }
        const Tree& get() const { return at->operator[](i); }
    };

    py::class_<TreeRef>(m, "Tree")
        .def(py::init<>())
        .def("root", [](const TreeRef& r) { return r.get().root().id(); })
        .def("num_leafs", [](const TreeRef& r) { return r.get().num_leafs(); })
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
        .def("get_split", [](const TreeRef& r, NodeId n) { return r.get()[n].get_split(); })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, FloatT v) { r.get()[n].set_leaf_value(v); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, FloatT sv) { r.get()[n].split({fid, sv}); })
        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); })
        ;

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        .def("__getitem__", [](std::shared_ptr<AddTree> at, size_t i) {
                if (i < at->size())
                    return TreeRef{at, i};
                throw py::value_error("out of bounds access into AddTree");
            })
        .def("__len__", &AddTree::size)
        .def("__iter__", [](const AddTree &at) { return py::make_iterator(at.begin(), at.end()); },
                    py::keep_alive<0, 1>())
        .def("num_nodes", &AddTree::num_nodes)
        .def("num_leafs", &AddTree::num_leafs)
        .def("get_splits", &AddTree::get_splits)
        .def("prune", &AddTree::prune)
        .def("__str__", [](const AddTree& at) { return tostr(at); })
    //    .def(py::pickle(
    //        [](const AddTree& at) { // __getstate__
    //            return at.to_json();
    //        },
    //        [](const std::string& json) { // __setstate__
    //            return AddTree::from_json(json);
    //        }))
    //    .def("get_domains", [](const AddTree& at, std::vector<NodeId> leaf_ids) {
    //        if (at.size() != leaf_ids.size())
    //            throw std::runtime_error("one leaf_id per tree in AddTree");

    //        DomainsT domains;
    //        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index)
    //        {
    //            NodeId leaf_id = leaf_ids[tree_index];
    //            const auto& tree = at[tree_index];
    //            auto node = tree[leaf_id];
    //            if (!node.is_leaf())
    //                throw std::runtime_error("leaf_id does not point to leaf");
    //            node.get_domains(domains);
    //        }
    //        return domains;
    //    })
        ;


} /* PYBIND11_MODULE */
