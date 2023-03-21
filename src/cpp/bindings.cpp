/*
 * Copyright 2022 DTAI Research Group - KU Leuven.
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
#include <pybind11/iostream.h>

#include "interval.hpp"
#include "features.hpp"
#include "json_io.hpp"
#include "tree.hpp"
#include "fp_search.hpp"

namespace py = pybind11;
using namespace veritas;

template <typename T>
std::string tostr(const T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

/** Convert between Python list and C++ Veritas box. */
Box::BufT tobox(py::object pybox) {
    Box::BufT buf;
    Box box{buf};
    FeatId count = 0;

    for (const auto& x : pybox) {
        Interval d;
        if (py::isinstance<py::tuple>(x)) {
            py::tuple t = py::cast<py::tuple>(x);
            FeatId id = py::cast<FeatId>(t[0]);
            d = py::cast<Interval>(t[1]);
            count = id;
        }
        else if (py::isinstance<py::int_>(x)) { // iterating dict
            FeatId id = py::cast<FeatId>(x);
            d = py::cast<Interval>(pybox[x]);
            count = id;
        }
        else if (py::isinstance<Interval>(x)) {
            d = py::cast<Interval>(x);
        }

        if (!box.refine_box(count, d))
            throw std::runtime_error("invalid box");

        //for (auto bb : box)//debug print
        //{
        //    if (bb.feat_id == count)
        //    {
        //        std::cout << "- in box:   " << bb.domain << " equal? "
        //            << (bb.domain.lo == d.lo) << (bb.domain.hi == d.hi) << std::endl
        //            << "  in pybox: " << d << std::endl;
        //    }
        //}
        ++count;
    }
    return buf;
}

data<FloatT>
get_data(py::handle h) {
    auto arr = py::array::ensure(h);
    if (!arr) throw std::runtime_error("invalid eval array");
    if (!arr.dtype().is(pybind11::dtype::of<FloatT>()))
        throw std::runtime_error("invalid dtype");

    py::buffer_info buf = arr.request();
    data d { static_cast<FloatT *>(buf.ptr), 0, 0, 0, 0 };
    if (buf.ndim == 1) {
        d.num_rows = 1;
        d.num_cols = buf.shape[0];
        d.stride_row = 0; // there is only one row
        d.stride_col = buf.strides[0] / sizeof(FloatT);
    } else if (buf.ndim == 2) {
        d.num_rows = buf.shape[0];
        d.num_cols = buf.shape[1];
        d.stride_row = buf.strides[0] / sizeof(FloatT);
        d.stride_col = buf.strides[1] / sizeof(FloatT);
    }
    else throw py::value_error("invalid data");
    return d;
}

PYBIND11_MODULE(veritas_core, m) {
    m.doc() = "Veritas: verification of tree ensembles";

    // redirect C++ output to Pythons stdout
    // https://github.com/pybind/pybind11/issues/1005
    // https://github.com/pybind/pybind11/pull/1009
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html#capturing-standard-output-from-ostream
    m.attr("_redirect_output") = py::capsule(
            new py::scoped_ostream_redirect(
                std::cout, py::module::import("sys").attr("stdout")),
            [](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });

    py::class_<Interval>(m, "Interval")
        .def(py::init<>())
        .def(py::init<FloatT, FloatT>())
        .def_static("from_lo", &Interval::from_lo)
        .def_static("from_hi", &Interval::from_hi)
        .def_readwrite("lo", &Interval::lo)
        .def_readwrite("hi", &Interval::hi)
        .def("lo_is_unbound", &Interval::lo_is_unbound)
        .def("hi_is_unbound", &Interval::hi_is_unbound)
        .def("contains", &Interval::contains)
        .def("overlaps", &Interval::overlaps)
        .def("intersect", &Interval::intersect)
        .def("is_everything", &Interval::is_everything)
        .def("split", &Interval::split)
        .def("__eq__", [](const Interval& s, const Interval& t) { return s == t; })
        .def("__repr__", [](const Interval& d) { return tostr(d); })
        .def("__iter__", [](const Interval& d) { return py::iter(py::make_tuple(d.lo, d.hi)); })
        .def(py::pickle(
            [](const Interval& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return Interval(t[0].cast<FloatT>(), t[1].cast<FloatT>());
            }))
        ; // Interval

    m.attr("BOOL_SPLIT_VALUE") = BOOL_SPLIT_VALUE;
    m.attr("TRUE_DOMAIN") = TRUE_DOMAIN;
    m.attr("FALSE_DOMAIN") = FALSE_DOMAIN;

    py::class_<IntervalPair>(m, "IntervalPair")
        .def_readonly("feat_id", &IntervalPair::feat_id)
        .def_readonly("interval", &IntervalPair::interval)
        ; // IntervalPair

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
        ; // LtSplit


    /* Avoid invalid pointers to Tree's by storing indexes rather than pointers */
    struct TreeRef {
        std::shared_ptr<AddTree> at;
        size_t i;
        Tree& get() { return at->operator[](i); }
        const Tree& get() const { return at->operator[](i); }
    };

    py::class_<TreeRef>(m, "Tree")
        .def("root", [](const TreeRef& r) { return r.get().root(); })
        .def("num_leaves", [](const TreeRef& r) { return r.get().num_leaves(); })
        .def("num_nodes", [](const TreeRef& r) { return r.get().num_nodes(); })
        .def("is_root", [](const TreeRef& r, NodeId n) { return r.get().is_root(n); })
        .def("is_leaf", [](const TreeRef& r, NodeId n) { return r.get().is_leaf(n); })
        .def("is_internal", [](const TreeRef& r, NodeId n) { return r.get().is_internal(n); })
        .def("left", [](const TreeRef& r, NodeId n) { return r.get().left(n); })
        .def("right", [](const TreeRef& r, NodeId n) { return r.get().right(n); })
        .def("parent", [](const TreeRef& r, NodeId n) { return r.get().parent(n); })
        .def("tree_size", [](const TreeRef& r, NodeId n) { return r.get().tree_size(n); })
        .def("depth", [](const TreeRef& r, NodeId n) { return r.get().depth(n); })
        .def("get_leaf_value", [](const TreeRef& r, NodeId n) { return r.get().leaf_value(n); })
        .def("get_split", [](const TreeRef& r, NodeId n) { return r.get().get_split(n); })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, FloatT v) { r.get().leaf_value(n) = v; })
        .def("find_minmax_leaf_value", [](const TreeRef& r, NodeId n)
                { return r.get().find_minmax_leaf_value(n); })
        .def("get_leaf_ids", [](const TreeRef& r) { return r.get().get_leaf_ids(); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, FloatT sv) { r.get().split(n, {fid, sv}); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid) { r.get().split(n, bool_ltsplit(fid)); })
        .def("eval", [](const TreeRef& r, py::handle arr, NodeId nid) {
            data d = get_data(arr);

            auto result = py::array_t<FloatT>(d.num_rows);
            py::buffer_info out = result.request();
            FloatT *out_ptr = static_cast<FloatT *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = r.get().eval(nid, d.row(i));

            return result;
        })
        .def("eval_node", [](const TreeRef& r, py::handle arr, NodeId nid) {
            data d = get_data(arr);

            auto result = py::array_t<NodeId>(d.num_rows);
            py::buffer_info out = result.request();
            NodeId *out_ptr = static_cast<NodeId *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = r.get().eval_node(nid, d.row(i));

            return result;
        })

        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); })
        .def("compute_box", [](const TreeRef& r, NodeId n) {
            Box::BufT buf;
            Box box{buf};
            r.get().compute_box(n, box);
            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("prune", [](const TreeRef& r, const py::object& pybox) {
            Box::BufT buf = tobox(pybox);
            Box box{buf};
            AddTree at;
            at.add_tree(r.get().prune(BoxRef{box}));
            return at;
        })
        ; // TreeRef

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        //.def(py::init<const AddTree&, size_t, size_t>())
        .def_readwrite("base_score", &AddTree::base_score)
        .def("copy", [](const AddTree& at) { return AddTree(at); })
        .def("__getitem__", [](const std::shared_ptr<AddTree>& at, size_t i) {
                if (i < at->size())
                    return TreeRef{at, i};
                throw py::value_error("out of bounds access into AddTree");
            })
        .def("__len__", &AddTree::size)
        .def("num_nodes", &AddTree::num_nodes)
        .def("num_leafs", &AddTree::num_leafs)
        .def("get_splits", &AddTree::get_splits)
        .def("add_tree", [](const std::shared_ptr<AddTree>& at) {
                at->add_tree(); return TreeRef{at, at->size()-1}; })
        .def("add_tree", [](const std::shared_ptr<AddTree>& at, const TreeRef& tref) {
                at->add_tree(tref.get()); // copy
                return TreeRef{at, at->size()-1}; })
        .def("prune", [](AddTree& at, const py::object& pybox) {
            Box::BufT buf = tobox(pybox);
            Box box{buf};
            return at.prune(BoxRef{box});
        })
        .def("neutralize_negative_leaf_values", &AddTree::neutralize_negative_leaf_values)
        .def("negate_leaf_values", &AddTree::negate_leaf_values)
        .def("concat_negated", &AddTree::concat_negated)
        .def("to_json", [](const AddTree& at) {
            std::stringstream ss;
            addtree_to_json(ss, at);
            return ss.str();
        })
        .def_static("from_json", [](const std::string& json) {
            std::stringstream s(json);
            return addtree_from_json<AddTree>(s);
        })
        .def_static("from_oldjson", [](const std::string& json) {
            std::stringstream s(json);
            return addtree_from_oldjson(s);
        })
        .def("eval", [](const AddTree& at, py::handle arr) {
            data d = get_data(arr);

            auto result = py::array_t<FloatT>(d.num_rows);
            py::buffer_info out = result.request();
            FloatT *out_ptr = static_cast<FloatT *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = at.eval(d.row(i));

            return result;
        })
        .def("compute_box", [](const AddTree& at, const std::vector<NodeId>& leaf_ids) {
            if (at.size() != leaf_ids.size())
                throw std::runtime_error("one leaf_id per tree in AddTree");

            Box::BufT buf;
            Box box{buf};
            at.compute_box(box, leaf_ids);

            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("__str__", [](const AddTree& at) { return tostr(at); })
        .def(py::pickle(
            [](const AddTree& at) { // __getstate__
                std::stringstream s;
                addtree_to_json(s, at);
                return s.str();
            },
            [](const std::string& json) { // __setstate__
                AddTree at;
                std::stringstream s(json);
                return addtree_from_json<AddTree>(s);
                return at;
            }))
        ; // AddTree

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
        ; // FeatMap

    py::enum_<StopReason>(m, "StopReason")
        .value("NONE", StopReason::NONE)
        .value("NO_MORE_OPEN", StopReason::NO_MORE_OPEN)
        .value("NUM_SOLUTIONS_EXCEEDED", StopReason::NUM_SOLUTIONS_EXCEEDED)
        .value("NUM_NEW_SOLUTIONS_EXCEEDED", StopReason::NUM_NEW_SOLUTIONS_EXCEEDED)
        .value("OPTIMAL", StopReason::OPTIMAL)
        .value("ATLEAST_BOUND_BETTER_THAN", StopReason::ATLEAST_BOUND_BETTER_THAN)
        .value("OUT_OF_TIME", StopReason::OUT_OF_TIME)
        .value("OUT_OF_MEMORY", StopReason::OUT_OF_MEMORY)
        ; // StopReason
    
    py::class_<Bounds>(m, "Bounds")
        .def_readonly("atleast", &Bounds::atleast)
        .def_readonly("best", &Bounds::best)
        .def_readonly("top_of_open", &Bounds::top_of_open)
        ; // Bounds
    
    py::class_<Statistics>(m, "Statistics")
        .def_readonly("num_steps", &Statistics::num_steps)
        .def_readonly("num_states_ignored", &Statistics::num_states_ignored)
        ; // Statistics

    py::class_<Settings>(m, "Settings")
        .def_readwrite("focal_eps", &Settings::focal_eps)
        .def_readwrite("max_focal_size", &Settings::max_focal_size)
        .def_readwrite("stop_when_num_solutions_exceeds",
                &Settings::stop_when_num_solutions_exceeds)
        .def_readwrite("stop_when_num_new_solutions_exceeds",
                &Settings::stop_when_num_new_solutions_exceeds)
        .def_readwrite("stop_when_optimal",
                &Settings::stop_when_optimal)
        .def_readwrite("ignore_state_when_worse_than",
                &Settings::ignore_state_when_worse_than)
        .def_readwrite("stop_when_atleast_bound_better_than",
                &Settings::stop_when_atleast_bound_better_than)
        ; // Settings

    py::class_<Search, std::shared_ptr<Search>>(m, "Search")
        .def_static("max_output", [](const AddTree& at, const py::object& pybox) {
            auto buf = tobox(pybox);
            auto fbox = BoxRef(buf).to_flatbox();
            return Search::max_output(at, fbox);
        }, py::arg("at"), py::arg("prune_box") = py::list())
        .def_static("min_output", [](const AddTree& at, const py::object& pybox) {
            auto buf = tobox(pybox);
            auto fbox = BoxRef(buf).to_flatbox();
            return Search::min_output(at, fbox);
        }, py::arg("at"), py::arg("prune_box") = py::list())
        .def("step", &Search::step)
        .def("steps", &Search::steps)
        .def("step_for", &Search::step_for)
        .def("num_solutions", &Search::num_solutions)
        .def("num_open", &Search::num_open)
        .def("get_max_memory", &Search::get_max_memory)
        .def("set_max_memory", &Search::set_max_memory)
        .def("get_used_memory", &Search::get_used_memory)
        .def("time_since_start", &Search::time_since_start)
        .def("current_bounds", &Search::current_bounds)
        .def("get_solution", &Search::get_solution)
        .def("get_solution_nodes", &Search::get_solution_nodes)
        .def("is_optimal", &Search::is_optimal)
//        .def("get_at_output_for_box", [](const VSearch& s, const py::list& pybox) {
//            Box box = tobox(pybox);
//            BoxRef b(box);
//            return s.get_at_output_for_box(b);
//        })
        .def_readonly("stats", &Search::stats)
        .def_readwrite("settings", &Search::settings)
        ; // Search

    py::class_<Solution>(m, "Solution")
        //.def_readonly("eps", &Solution::eps)
        .def_readonly("time", &Solution::time)
        .def_readonly("output", &Solution::output)
        .def("box", [](const Solution& s) {
            py::dict d;
            for (auto&& [feat_id, ival] : s.box)
                d[py::int_(feat_id)] = ival;
            return d;
        })
        .def("__str__", [](const Solution& s) { return tostr(s); })
        ; // Solution



} /* PYBIND11_MODULE */
