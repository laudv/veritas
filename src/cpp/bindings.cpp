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

#include "domain.hpp"
#include "features.hpp"
#include "tree.hpp"
//#include "graph_search.hpp"
#include "search.hpp"
#include "constraints.hpp"

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
Box
tobox(py::object pybox)
{
    Box box;
    FeatId count = 0;

    for (const auto& x : pybox)
    {
        Domain d;
        if (py::isinstance<py::tuple>(x))
        {
            py::tuple t = py::cast<py::tuple>(x);
            FeatId id = py::cast<FeatId>(t[0]);
            d = py::cast<Domain>(t[1]);
            count = id;
        }
        else if (py::isinstance<py::int_>(x)) // iterating dict
        {
            FeatId id = py::cast<FeatId>(x);
            d = py::cast<Domain>(pybox[x]);
            count = id;
        }
        else if (py::isinstance<Domain>(x))
        {
            d = py::cast<Domain>(x);
        }

        if (!refine_box(box, count, d))
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
    return box;
}

data
get_data(py::handle h)
{
    auto arr = py::array::ensure(h);
    if (!arr) throw std::runtime_error("invalid eval array");
    if (!arr.dtype().is(pybind11::dtype::of<FloatT>()))
        throw std::runtime_error("invalid dtype");

    py::buffer_info buf = arr.request();
    data d { static_cast<FloatT *>(buf.ptr), 0, 0, 0, 0 };
    if (buf.ndim == 1)
    {
        d.num_rows = 1;
        d.num_cols = buf.shape[0];
        d.stride_row = 0; // there is only one row
        d.stride_col = buf.strides[0] / sizeof(FloatT);
    }
    else if (buf.ndim == 2)
    {
        d.num_rows = buf.shape[0];
        d.num_cols = buf.shape[1];
        d.stride_row = buf.strides[0] / sizeof(FloatT);
        d.stride_col = buf.strides[1] / sizeof(FloatT);
    }
    else throw py::value_error("invalid data");
    return d;
}

PYBIND11_MODULE(pyveritas, m) {
    m.doc() = "Veritas: verification of tree ensembles";

    // redirect C++ output to Pythons stdout
    // https://github.com/pybind/pybind11/issues/1005
    // https://github.com/pybind/pybind11/pull/1009
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html#capturing-standard-output-from-ostream
    m.attr("_redirect_output") = py::capsule(
            new py::scoped_ostream_redirect(
                std::cout, py::module::import("sys").attr("stdout")),
            [](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });

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
        .def("intersect", &Domain::intersect)
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
        ; // Domain

    m.attr("BOOL_SPLIT_VALUE") = BOOL_SPLIT_VALUE;
    m.attr("TRUE_DOMAIN") = TRUE_DOMAIN;
    m.attr("FALSE_DOMAIN") = FALSE_DOMAIN;

    py::class_<DomainPair>(m, "DomainPair")
        .def_readonly("feat_id", &DomainPair::feat_id)
        .def_readonly("domain", &DomainPair::domain)
        ; // DomainPair

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
        .def("find_minmax_leaf_value", [](const TreeRef& r, NodeId n)
                { return r.get()[n].find_minmax_leaf_value(); })
        .def("get_leaf_ids", [](const TreeRef& r) { return r.get().get_leaf_ids(); })
        .def("leaf_value_variance", [](TreeRef& r) { return r.get().leaf_value_variance(); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, FloatT sv) { r.get()[n].split({fid, sv}); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid) { r.get()[n].split(fid); })
        .def("eval", [](const TreeRef& r, py::handle arr) {
            data d = get_data(arr);

            auto result = py::array_t<FloatT>(d.num_rows);
            py::buffer_info out = result.request();
            FloatT *out_ptr = static_cast<FloatT *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = r.get().eval(d.row(i));

            return result;
        })
        .def("eval_node", [](const TreeRef& r, py::handle arr) {
            data d = get_data(arr);

            auto result = py::array_t<NodeId>(d.num_rows);
            py::buffer_info out = result.request();
            NodeId *out_ptr = static_cast<NodeId *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = r.get().eval_node(d.row(i));

            return result;
        })
        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); })
        .def("compute_box", [](const TreeRef& r, NodeId n) {
            Box box = r.get()[n].compute_box();
            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("prune", [](const TreeRef& r, const py::object& pybox) {
            Box box = tobox(pybox);
            BoxRef b(box);
            AddTree at;
            Tree t = r.get().prune(b);
            at.add_tree(std::move(t));
            return at;
        })
        ; // TreeRef

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<>())
        .def(py::init<const AddTree&, size_t, size_t>())
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
            Box box = tobox(pybox);
            BoxRef b(box);
            //py::print("pruning AddTree using box", tostr(b));
            return at.prune(b);
        })
        .def("neutralize_negative_leaf_values", &AddTree::neutralize_negative_leaf_values)
        .def("negate_leaf_values", &AddTree::negate_leaf_values)
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

            Box box;
            for (size_t tree_index = 0; tree_index < at.size(); ++tree_index)
            {
                NodeId leaf_id = leaf_ids[tree_index];
                const Tree& tree = at[tree_index];
                auto node = tree[leaf_id];
                if (!node.is_leaf())
                    throw std::runtime_error("leaf_id does not point to leaf");
                bool success = node.compute_box(box);
                if (!success)
                    throw std::runtime_error("non-overlapping leafs");
            }

            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("concat_negated", &AddTree::concat_negated)
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
        .value("UPPER_LT", StopReason::UPPER_LT)
        .value("LOWER_GT", StopReason::LOWER_GT)
        ; // StopReason


    py::class_<VSearch, std::shared_ptr<VSearch>>(m, "Search")
        .def_static("max_output", &VSearch::max_output)
        .def_static("min_dist_to_example", &VSearch::min_dist_to_example)
        .def("step", &VSearch::step)
        .def("steps", &VSearch::steps)
        .def("step_for", &VSearch::step_for)
        .def("steps", &VSearch::steps)
        .def("step_for", &VSearch::step_for)
        .def("num_solutions", &VSearch::num_solutions)
        .def("num_open", &VSearch::num_open)
        .def("set_mem_capacity", &VSearch::set_mem_capacity)
        .def("remaining_mem_capacity", &VSearch::remaining_mem_capacity)
        .def("used_mem_size", &VSearch::used_mem_size)
        .def("time_since_start", &VSearch::time_since_start)
        .def("current_bounds", &VSearch::current_bounds)
        .def("get_solution", &VSearch::get_solution)
        .def("get_solution_nodes", &VSearch::get_solution_nodes)
        .def("is_optimal", &VSearch::is_optimal)
        .def("base_score", &VSearch::base_score)
        .def("get_at_output_for_box", [](const VSearch& s, const py::list& pybox) {
            Box box = tobox(pybox);
            BoxRef b(box);
            return s.get_at_output_for_box(b);
        })
        .def("prune", [](VSearch& s, const py::object& pybox) {
            Box box = tobox(pybox);
            BoxRef b(box);
            return s.prune_by_box(b);
        })
        .def("get_solstate_field", [](const VSearch& s, size_t index, const std::string& field) -> py::object {
            if (const auto* v = dynamic_cast<const Search<MaxOutputHeuristic>*>(&s))
            {
                const auto& s = v->get_solution_state(index);
                if (field == "g")
                    return py::float_(s.g);
                if (field == "h")
                    return py::float_(s.h);
            }
            else if(const auto* v = dynamic_cast<const Search<MinDistToExampleHeuristic>*>(&s))
            {
                const auto& s = v->get_solution_state(index);
                if (field == "g")
                    return py::float_(s.g);
                if (field == "h")
                    return py::float_(s.h);
                if (field == "dist")
                    return py::float_(s.dist);
            }
            else {
                throw std::runtime_error("unsupported VSearch subtype");
            }
            return py::none();
        })

        .def("add_onehot_constraint", [](std::shared_ptr<VSearch> s, const std::vector<FeatId>& feat_ids) {
            if (auto* v = dynamic_cast<Search<MaxOutputHeuristic>*>(s.get()))
            {
                constraints::onehot(*v, feat_ids);
            }
            else {
                throw std::runtime_error("unsupported VSearch subtype");
            }
            return py::none();

            })

        .def("add_sqdist1_constraint", [](std::shared_ptr<VSearch> s, FeatId x, FeatId y, FeatId d, FloatT x0, FloatT y0) {
            if (auto* v = dynamic_cast<Search<MaxOutputHeuristic>*>(s.get()))
            {
                constraints::sqdist1(*v, x, y, d, x0, y0);
            }
            else {
                throw std::runtime_error("unsupported VSearch subtype");
            }
            return py::none();

            })



        // stats
        .def_readonly("num_steps", &VSearch::num_steps)
        .def_readonly("num_rejected_solutions", &VSearch::num_rejected_solutions)
        .def_readonly("num_rejected_states", &VSearch::num_rejected_states)
        .def_readonly("snapshots", &VSearch::snapshots)

        // options
        .def_readwrite("eps", &VSearch::eps)
        .def_readwrite("debug", &VSearch::debug)
        .def_readwrite("max_focal_size", &VSearch::max_focal_size)
        .def_readwrite("auto_eps", &VSearch::auto_eps)
        .def_readwrite("reject_solution_when_output_less_than", &VSearch::reject_solution_when_output_less_than)

        // stop condition
        .def_readwrite("stop_when_num_solutions_exceeds",     &VSearch::stop_when_num_solutions_exceeds)
        .def_readwrite("stop_when_num_new_solutions_exceeds", &VSearch::stop_when_num_new_solutions_exceeds)
        .def_readwrite("stop_when_optimal",                   &VSearch::stop_when_optimal)
        .def_readwrite("stop_when_upper_less_than",           &VSearch::stop_when_upper_less_than)
        .def_readwrite("stop_when_lower_greater_than",        &VSearch::stop_when_lower_greater_than)
        ; // VSearch

    py::class_<Solution>(m, "Solution")
        .def_readonly("eps", &Solution::eps)
        .def_readonly("time", &Solution::time)
        .def_readonly("output", &Solution::output)
        .def("box", [](const Solution& s) {
            py::dict d;
            for (auto&& [feat_id, dom] : s.box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("__str__", [](const Solution& s) { return tostr(s); })
        ; // Solution

    py::class_<Snapshot>(m, "Snapshot")
        .def_readonly("time", &Snapshot::time)
        .def_readonly("num_steps", &Snapshot::num_steps)
        .def_readonly("num_solutions", &Snapshot::num_solutions)
        .def_readonly("num_open", &Snapshot::num_open)
        .def_readonly("eps", &Snapshot::eps)
        .def_readonly("bounds", &Snapshot::bounds)
        .def_readonly("avg_focal_size", &Snapshot::avg_focal_size)
        ; // Snapshot



} /* PYBIND11_MODULE */
