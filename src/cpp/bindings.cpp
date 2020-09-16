/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
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

#include <z3++.h>

#include "domain.h"
#include "tree.h"
#include "graph.h"
#include "smt.h"

namespace py = pybind11;
using namespace veritas;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

static AddTree DUMMY_ADDTREE{};

using TreeD = Tree<Split, FloatT>;
using NodeRefD = TreeD::MRef;

PYBIND11_MODULE(pyveritas, m) {
    m.doc() = "Tree-CK: verification of ensembles of trees";

    py::class_<RealDomain>(m, "RealDomain")
        .def(py::init<>())
        .def(py::init<FloatT, FloatT>())
        .def_readwrite("lo", &RealDomain::lo)
        .def_readwrite("hi", &RealDomain::hi)
        .def("lo_is_inf", &RealDomain::lo_is_inf)
        .def("hi_is_inf", &RealDomain::hi_is_inf)
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

    py::class_<LtSplit>(m, "LtSplit")
        .def(py::init<FeatId, LtSplit::ValueT>())
        .def_readonly("feat_id", &LtSplit::feat_id)
        .def_readonly("split_value", &LtSplit::split_value)
        .def("test", &LtSplit::test)
        .def("__eq__", [](const LtSplit& s, const LtSplit t) { return s == t; })
        .def("__repr__", [](const LtSplit& s) { return tostr(s); })
        .def(py::pickle(
            [](const LtSplit& s) { return py::make_tuple(s.feat_id, s.split_value); }, // __getstate__
            [](py::tuple t) -> LtSplit { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return { t[0].cast<FeatId>(), t[1].cast<FloatT>() };
            }));

    py::class_<BoolSplit>(m, "BoolSplit")
        .def(py::init<FeatId>())
        .def_readonly("feat_id", &BoolSplit::feat_id)
        .def("test", &BoolSplit::test)
        .def("__eq__", [](const BoolSplit& s, const BoolSplit t) { return s == t; })
        .def("__repr__", [](const BoolSplit& s) { return tostr(s); })
        .def(py::pickle(
            [](const BoolSplit& s) { return py::make_tuple(s.feat_id); }, // __getstate__
            [](py::tuple t) -> BoolSplit { // __setstate__
                if (t.size() != 1) throw std::runtime_error("invalid pickle state");
                return { t[0].cast<FeatId>() };
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
        .def("num_leafs", [](const TreeRef& r) { return r.get().num_leafs(); })
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
        .def("get_domains", [](const AddTree& at, std::vector<NodeId> leaf_ids) {
            if (at.size() != leaf_ids.size())
                throw std::runtime_error("one leaf_id per tree in AddTree");

            DomainsT domains;
            for (size_t tree_index = 0; tree_index < at.size(); ++tree_index)
            {
                NodeId leaf_id = leaf_ids[tree_index];
                const auto& tree = at[tree_index];
                auto node = tree[leaf_id];
                if (!node.is_leaf())
                    throw std::runtime_error("leaf_id does not point to leaf");
                node.get_domains(domains);
            }
            return domains;
        })
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

    py::class_<FeatInfo>(m, "FeatInfo")
        .def(py::init<const AddTree&, const AddTree&, const std::unordered_set<FeatId>&, bool>())
        .def("get_id", &FeatInfo::get_id)
        .def("is_instance0_id", &FeatInfo::is_instance0_id)
        .def("is_real", &FeatInfo::is_real)
        .def("feat_ids0", &FeatInfo::feat_ids0)
        .def("feat_ids1", &FeatInfo::feat_ids1)
        ;

    py::class_<KPartiteGraph>(m, "KPartiteGraph")
        .def(py::init<const AddTree&, const FeatInfo&, int>())
        .def("set_max_mem_size", [](KPartiteGraph& g, size_t m) { g.store().set_max_mem_size(m); })
        .def("get_max_mem_size", [](const KPartiteGraph& o) { return o.store().get_max_mem_size(); })
        .def("get_mem_size", [](const KPartiteGraph& o) { return o.store().get_mem_size(); })
        .def("get_used_mem_size", [](const KPartiteGraph& o) { return o.store().get_used_mem_size(); })
        .def("merge", &KPartiteGraph::merge)
        .def("basic_bound", &KPartiteGraph::basic_bound)
        .def("num_independent_sets", &KPartiteGraph::num_independent_sets)
        .def("num_vertices", &KPartiteGraph::num_vertices)
        .def("num_vertices_in_set", &KPartiteGraph::num_vertices_in_set)
        .def("__str__", [](const KPartiteGraph& g) { return tostr(g); })
        .def("prune_example", [](KPartiteGraph& g, const FeatInfo& finfo, const py::list& example, FloatT delta) {
            for (FeatId fid : finfo.feat_ids0())
            {
                const py::handle& o = example[fid];
                if (py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o))
                {
                    FloatT v = o.cast<FloatT>();
                    auto f = [=](FeatId i) { return finfo.get_id(0, i); };
                    g.store().refine_workspace(LtSplit(fid, v-delta), false, f);
                    g.store().refine_workspace(LtSplit(fid, v+delta), true, f);
                }
                //else throw std::runtime_error("not supported");
            }
            for (FeatId fid : finfo.feat_ids1())
            {
                const py::handle& o = example[fid];
                if (py::isinstance<py::float_>(o))
                {
                    FloatT v = o.cast<FloatT>();
                    auto f = [=](FeatId i) { return finfo.get_id(1, i); };
                    g.store().refine_workspace(LtSplit(fid, v-delta), false, f);
                    g.store().refine_workspace(LtSplit(fid, v+delta), true, f);
                }
                //else throw std::runtime_error("not supported");
            }

            DomainBox b = g.store().get_workspace_box();
            g.prune([b](const DomainBox& box) {
                return box.overlaps(b);
            });
            g.store().clear_workspace();
        })
        .def("prune_box", [](KPartiteGraph& g, const FeatInfo& finfo,
                    const py::list box, int instance) {
            for (FeatId fid : instance == 0 ? finfo.feat_ids0() : finfo.feat_ids1())
            {
                RealDomain d = box[fid].cast<RealDomain>();
                auto f = [=](FeatId i) { return finfo.get_id(instance, i); };
                if (!std::isinf(d.lo))
                    g.store().refine_workspace(LtSplit(fid, d.lo), false, f);
                if (!std::isinf(d.hi))
                    g.store().refine_workspace(LtSplit(fid, d.hi), true, f);
            }
            DomainBox b = g.store().get_workspace_box();
            g.prune([b](const DomainBox& box) {
                return box.overlaps(b);
            });
            g.store().clear_workspace();
        })
        .def("prune_smt", [](KPartiteGraph& g, SMTSolver& solver) {
            g.prune([&solver](const DomainBox& box) {
                z3::expr e = solver.domains_to_z3(box);
                bool res = solver.check(e);
                //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
                return res;
            });
        })
        .def("add_with_negated_leaf_values", &KPartiteGraph::add_with_negated_leaf_values)
        ;

    py::class_<Solution>(m, "Solution")
        .def_readonly("output0", &Solution::output0)
        .def_readonly("output1", &Solution::output1)
        .def_readonly("eps", &Solution::eps)
        .def_readonly("time", &Solution::time)
        .def_readonly("is_valid", &Solution::is_valid)
        .def("output_difference", &Solution::output_difference)
        .def("box", [](const Solution& sol) {
            py::dict b;
            for (auto&& [id, d] : sol.box)
                b[py::int_(id)] = d;
            return b;
        })
        .def("__str__", [](const Solution& s) { return tostr(s); })
        ;

    py::enum_<KPartiteGraphOptimize::Heuristic>(m, "KPartiteGraphOptimizeHeuristic")
        .value("DYN_PROG", KPartiteGraphOptimize::Heuristic::DYN_PROG)
        .value("RECOMPUTE", KPartiteGraphOptimize::Heuristic::RECOMPUTE)
        .export_values();

    py::class_<KPartiteGraphOptimize>(m, "KPartiteGraphOptimize")
        .def(py::init<KPartiteGraph&, KPartiteGraph&, KPartiteGraphOptimize::Heuristic>(),
                py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("set_max_mem_size", [](KPartiteGraphOptimize& o, size_t m) { o.store().set_max_mem_size(m); })
        .def("get_max_mem_size", [](const KPartiteGraphOptimize& o) { return o.store().get_max_mem_size(); })
        .def("get_mem_size", [](const KPartiteGraphOptimize& o) { return o.store().get_mem_size(); })
        .def("get_used_mem_size", [](const KPartiteGraphOptimize& o) { return o.store().get_used_mem_size(); })
        .def_readonly("num_steps", &KPartiteGraphOptimize::num_steps)
        .def_readonly("num_update_fails", &KPartiteGraphOptimize::num_update_fails)
        .def_readonly("num_rejected", &KPartiteGraphOptimize::num_rejected)
        .def_readonly("num_box_checks", &KPartiteGraphOptimize::num_box_checks)
        .def_readonly("solutions", &KPartiteGraphOptimize::solutions)
        .def_readonly("start_time", &KPartiteGraphOptimize::start_time)
        .def("num_solutions", [](const KPartiteGraphOptimize& o) { return o.solutions.size(); })
        .def("current_bounds", &KPartiteGraphOptimize::current_bounds)
        .def("num_candidate_cliques", &KPartiteGraphOptimize::num_candidate_cliques)
        .def("get_eps", &KPartiteGraphOptimize::get_eps)
        .def("set_eps", [](KPartiteGraphOptimize& o, FloatT eps) { o.set_eps(eps); })
        //.def("__str__", [](const KPartiteGraphOptimize& o) { return tostr(o); })
        .def("steps", [](KPartiteGraphOptimize& opt, int nsteps, py::kwargs kwargs) {
            //auto f = [opt](const DomainBox& box) {
            //    z3::expr e = opt.solver->domains_to_z3(box);
            //    bool res = opt.solver->check(e);
            //    //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
            //    //std::cout << opt.solver->get_z3() << std::endl;
            //    return res;
            //};
            EasyBoxAdjuster eadj;
            if (kwargs.contains("adjuster"))
                eadj = kwargs["adjuster"].cast<EasyBoxAdjuster>();

            if (kwargs.contains("min_output_difference"))
            {
                FloatT min_output_difference = kwargs["min_output_difference"].cast<FloatT>();
                //if (opt.solver)
                //    return opt.opt->steps(nsteps, f, min_output_difference);
                return opt.steps(nsteps, eadj, min_output_difference);
            }
            else
            {
                FloatT max_output0 = std::numeric_limits<FloatT>::infinity();
                FloatT min_output1 = -std::numeric_limits<FloatT>::infinity();
                if (kwargs.contains("max_output"))
                    max_output0 = kwargs["max_output"].cast<FloatT>();
                if (kwargs.contains("min_output"))
                    min_output1 = kwargs["min_output"].cast<FloatT>();
                //if (opt.solver)
                //    return opt.opt->steps(nsteps, f, max_output0, min_output1);
                return opt.steps(nsteps, eadj, max_output0, min_output1);
            }
        })
        .def("parallel", [](const KPartiteGraphOptimize& opt, size_t num_threads) {
            return KPartiteGraphParOpt(num_threads, opt);
        })
        ;

    py::class_<KPartiteGraphParOpt>(m, "KPartiteGraphParOpt")
        .def("num_threads", &KPartiteGraphParOpt::num_threads)
        .def("num_solutions", &KPartiteGraphParOpt::num_solutions)
        .def("num_new_valid_solutions", [](const KPartiteGraphParOpt& o) { return o.num_new_valid_solutions(); })
        .def("num_candidate_cliques", &KPartiteGraphParOpt::num_candidate_cliques)
        .def("current_bounds", &KPartiteGraphParOpt::current_bounds)
        .def("current_memory", &KPartiteGraphParOpt::current_memory)
        .def("get_eps", &KPartiteGraphParOpt::get_eps)
        .def("set_eps", &KPartiteGraphParOpt::set_eps)
        .def("join_all", &KPartiteGraphParOpt::join_all)
        .def("worker_opt", &KPartiteGraphParOpt::worker_opt)
        .def("set_box_adjuster", [](KPartiteGraphParOpt& paropt, EasyBoxAdjuster adj) {
            paropt.set_box_adjuster([adj]() { return adj; }); // copy
        })
        .def("steps_for", [](KPartiteGraphParOpt& opt, size_t num_millisecs, py::kwargs kwargs) {
            if (kwargs.contains("min_output_difference"))
            {
                FloatT min_output_difference = kwargs["min_output_difference"].cast<FloatT>();
                opt.set_output_limits(min_output_difference);
            }
            else if (kwargs.contains("max_output"))
            {
                FloatT max_output0 = std::numeric_limits<FloatT>::infinity();
                FloatT min_output1 = -std::numeric_limits<FloatT>::infinity();
                if (kwargs.contains("max_output"))
                    max_output0 = kwargs["max_output"].cast<FloatT>();
                if (kwargs.contains("min_output"))
                    min_output1 = kwargs["min_output"].cast<FloatT>();
                opt.set_output_limits(max_output0, min_output1);
            }
            opt.steps_for(num_millisecs);
        });

    py::class_<SMTSolver>(m, "SMTSolver")
        .def(py::init<const FeatInfo *, const AddTree&, const AddTree&>())
        .def("parse_smt", &SMTSolver::parse_smt)
        .def("xvar_id", &SMTSolver::xvar_id)
        .def("xvar_name", &SMTSolver::xvar_name)
        ;


    py::class_<EasyBoxAdjuster>(m, "EasyBoxAdjuster")
        .def(py::init<>())
        .def("add_one_out_of_k", &EasyBoxAdjuster::add_one_out_of_k)
        .def("add_at_most_k", &EasyBoxAdjuster::add_at_most_k)
        .def("add_less_than", &EasyBoxAdjuster::add_less_than)
        ;


} /* PYBIND11_MODULE */
