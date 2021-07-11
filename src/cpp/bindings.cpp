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
#include <pybind11/numpy.h>

#include "domain.hpp"
#include "features.hpp"
#include "new_tree.hpp"
#include "search.hpp"

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
        .def("__iter__", [](const Domain& d) { return py::iter(py::make_tuple(d.lo, d.hi)); })
        .def(py::pickle(
            [](const Domain& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return Domain(t[0].cast<FloatT>(), t[1].cast<FloatT>());
            }))
        ;

    m.attr("BOOL_SPLIT_VALUE") = BOOL_SPLIT_VALUE;
    m.attr("TRUE_DOMAIN") = TRUE_DOMAIN;
    m.attr("FALSE_DOMAIN") = FALSE_DOMAIN;

    py::class_<DomainPair>(m, "DomainPair")
        .def_readonly("feat_id", &DomainPair::feat_id)
        .def_readonly("domain", &DomainPair::domain)
        ;

    py::class_<LtSplit>(m, "LtSplit")
        .def(py::init<FeatId, FloatT>())
        .def_readonly("feat_id", &LtSplit::feat_id)
        .def_readonly("split_value", &LtSplit::split_value)
        .def("test", [](const LtSplit& s, FloatT v) { return s.test(v); })
        .def("__eq__", [](const LtSplit& s, const LtSplit& t) { return s == t; })
        .def("__repr__", [](const LtSplit& s) { return tostr(s); })
        .def(py::pickle(
            [](const LtSplit& s) { return py::make_tuple(s.feat_id, s.split_value); }, // __getstate__
            [](py::tuple t) -> LtSplit { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return { t[0].cast<FeatId>(), t[1].cast<FloatT>() };
            }))
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
        .def("split", [](TreeRef& r, NodeId n, FeatId fid) { r.get()[n].split(fid); })
        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); })
        ;

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        .def(py::init<const AddTree&, size_t, size_t>())
        .def_readwrite("base_score", &AddTree::base_score)
        .def("copy", [](const AddTree& at) { return AddTree(at); })
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
        .def("add_tree", [](const std::shared_ptr<AddTree>& at) {
                at->add_tree(); return TreeRef{at, at->size()-1}; })
        .def("prune", [](AddTree& at, const py::list& pybox) {
            Box box;
            FeatId count = 0;
            for (const auto& x : pybox)
            {
                if (py::isinstance<py::tuple>(x))
                {
                    py::tuple t = py::cast<py::tuple>(x);
                    FeatId id = py::cast<FeatId>(t[0]);
                    Domain dom = py::cast<Domain>(t[1]);
                    box.push_back({id, dom});
                    count = id;
                }
                else if (py::isinstance<Domain>(x))
                {
                    Domain dom = py::cast<Domain>(x);
                    box.push_back({count, dom});
                }
                ++count;
            }
            BoxRef b(box);
            py::print("pruning using box", tostr(b));
            return at.prune(b);
        })
        .def("to_json", [](const AddTree& at) {
            std::stringstream s;
            at.to_json(s);
            return s.str();
        })
        .def_static("from_json", [](const std::string& json) {
            AddTree at;
            std::stringstream s(json);
            at.from_json(s);
            return at;
        })
        .def("eval", [](const AddTree& at, py::array_t<FloatT> arr) {
            py::buffer_info buf = arr.request();
            if (buf.ndim != 2) py::value_error("invalid data");

            std::cout << "eval: array strides: " << buf.strides[0] << ", " << buf.strides[1] << std::endl;

            row_major_data data {{ static_cast<FloatT *>(buf.ptr),
                                   static_cast<size_t>(buf.shape[0]),
                                   static_cast<size_t>(buf.shape[1]) }};

            auto result = py::array_t<FloatT>(buf.shape[0]);
            py::buffer_info out = result.request();
            FloatT *out_ptr = static_cast<FloatT *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(buf.shape[0]); ++i)
                out_ptr[i] = at.eval(row<row_major_data>{data, i});

            return result;
                

            //for (size_t i = 0; i < buf.shape[0]; ++i)
            //    for (size_t j = 0; j < buf.shape[1]; ++j)
            //        py::print(i, j, "=>", data.get_elem(i, j));
        })
        .def("compute_box", [](const AddTree& at, const std::vector<NodeId>& leaf_ids) {
            if (at.size() != leaf_ids.size())
                throw std::runtime_error("one leaf_id per tree in AddTree");

            Box box;
            for (size_t tree_index = 0; tree_index < at.size(); ++tree_index)
            {
                NodeId leaf_id = leaf_ids[tree_index];
                const Tree& tree = at[tree_index];
                auto node = tree[leaf_id];
                if (!node.is_leaf())
                    throw std::runtime_error("leaf_id does not point to leaf");
                node.compute_box(box);
            }

            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("__str__", [](const AddTree& at) { return tostr(at); })
        .def(py::pickle(
            [](const AddTree& at) { // __getstate__
                std::stringstream s;
                at.to_json(s);
                return s.str();
            },
            [](const std::string& json) { // __setstate__
                AddTree at;
                std::stringstream s(json);
                at.from_json(s);
                return at;
            }))
        ;

    py::class_<FeatMap>(m, "FeatMap")
        .def(py::init<FeatId>())
        .def(py::init<const std::vector<std::string>&>())
        .def("num_features", &FeatMap::num_features)
        .def("__len__", &FeatMap::num_features)
        .def("get_index", [](const FeatMap& fm, FeatId id, int inst) { return fm.get_index(id, inst); })
        .def("get_index", [](const FeatMap& fm, const std::string& n, int inst) { return fm.get_index(n, inst); })
        .def("get_instance", &FeatMap::get_instance)
        .def("get_name", &FeatMap::get_name)
        .def("get_indices_map", [](const FeatMap& fm) {
            auto map = fm.get_indices_map();
            py::object d = py::dict();
            for (auto&& [id, idx] : map)
            {
                if (d.contains(py::int_(id)))
                {
                    py::list l = d[py::int_(id)];
                    l.append(py::int_(idx));
                }
                else
                {
                    py::list l;
                    l.append(py::int_(idx));
                    d[py::int_(id)] = l;
                }
            }
            return d;
        })
        .def("share_all_features_between_instances", &FeatMap::share_all_features_between_instances)
        .def("get_feat_id", [](const FeatMap& fm, FeatId id) { return fm.get_feat_id(id); })
        .def("get_feat_id", [](const FeatMap& fm, const std::string& n, int i) { return fm.get_feat_id(n, i); })
        .def("use_same_id_for", &FeatMap::use_same_id_for)
        .def("transform", [](const FeatMap& fm, const AddTree& at, int i) { return fm.transform(at, i); })
        .def("__iter__", [](const FeatMap &fm) { return py::make_iterator(fm.begin(), fm.end()); },
                    py::keep_alive<0, 1>())
        .def("iter_instance", [](const FeatMap& fm, int i) {
                auto h = fm.iter_instance(i);
                return py::make_iterator(h.begin(), h.end()); },
                py::keep_alive<0, 1>())
        .def("__str__", [](const FeatMap& fm) { return tostr(fm); })
        ;

    py::class_<Search>(m, "Search")
        .def(py::init<const AddTree&>())
        .def("step", &Search::step)
        .def("steps", &Search::steps)
        .def("step_for", &Search::step_for)
        .def("num_solutions", &Search::num_solutions)
        .def("num_states", &Search::num_states)
        .def("get_solution", &Search::get_solution)
        .def("time_since_start", &Search::time_since_start)
        .def("current_bound", &Search::current_bound)
        .def("get_eps", &Search::get_eps)
        .def("set_eps", &Search::set_eps)
        .def_readwrite("max_mem_size", &Search::max_mem_size)
        .def_readonly("stats", &Search::stats)
        ;

    py::class_<Stats>(m, "Stats")
        .def_readonly("num_steps", &Stats::num_steps)
        .def_readonly("num_impossible", &Stats::num_impossible)
        .def_readonly("num_solutions", &Stats::num_solutions)
        .def_readonly("num_states", &Stats::num_states)
        .def_readonly("snapshots", &Stats::snapshots)
        ;

    py::class_<Snapshot>(m, "Snapshot")
        .def_readonly("num_steps", &Snapshot::num_steps)
        .def_readonly("num_impossible", &Snapshot::num_impossible)
        .def_readonly("num_solutions", &Snapshot::num_solutions)
        .def_readonly("num_states", &Snapshot::num_states)
        .def_readonly("time", &Snapshot::time)
        .def_readonly("bound", &Snapshot::bound)
        ;

    py::class_<Solution>(m, "Solution")
        .def_readonly("state_index", &Solution::state_index)
        .def_readonly("solution_index", &Solution::solution_index)
        .def_readonly("eps", &Solution::eps)
        .def_readonly("output", &Solution::output)
        .def_readonly("nodes", &Solution::nodes)
        .def_readonly("time", &Solution::time)
        .def("box", [](const Solution& s) {
            py::dict d;
            for (auto&& [feat_id, dom] : s.box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("__str__", [](const Solution& s) { return tostr(s); })
        ;


} /* PYBIND11_MODULE */
