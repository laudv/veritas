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
#include "domtree.h"
#include "graph.h"
#include "smt.h"

namespace py = pybind11;
using namespace treeck;

template <typename T>
std::string tostr(T& o)
{
    std::stringstream s;
    s << o;
    return s.str();
}

//static
//py::tuple
//encode_split(const DomTreeSplit& split)
//{
//    size_t i = split.instance_index;
//    return py::make_tuple(i, split.split);
//}

static AddTree DUMMY_ADDTREE{};

using TreeD = Tree<Split, FloatT>;
using NodeRefD = TreeD::MRef;
using DomTreeT = DomTree::DomTreeT;

PYBIND11_MODULE(pytreeck, m) {
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

        /*

    py::class_<DomTree>(m, "DomTree")
        .def(py::init<>())
        .def(py::init<>([](std::shared_ptr<AddTree> v, DomainsT d) -> DomTree {
            DomTree dt;
            dt.add_instance(v, std::move(d));
            return dt;
        }))
        .def(py::init<>([](std::vector<std::tuple<std::shared_ptr<AddTree>, DomainsT>>& v) -> DomTree {
            DomTree dt;
            for (auto& item : v)
                dt.add_instance(std::get<0>(item), std::move(std::get<1>(item)));
            return dt;
        }))
        .def("tree", &DomTree::tree)
        .def("addtree", &DomTree::addtree)
        .def("num_instances", &DomTree::num_instances)
        .def("get_root_domain", &DomTree::get_root_domain)
        .def("get_leaf", &DomTree::get_leaf)
        .def("apply_leaf", [](DomTree& dt, const DomTreeLeaf& leaf) {
            dt.apply_leaf(DomTreeLeaf { leaf });
        });

    py::class_<DomTreeLeaf>(m, "DomTreeLeaf")
        .def_readonly("score", &DomTreeLeaf::score)
        .def_readonly("balance", &DomTreeLeaf::balance)
        .def("addtree", &DomTreeLeaf::addtree)
        .def("domtree_leaf_id", &DomTreeLeaf::domtree_leaf_id)
        .def("num_instances", &DomTreeLeaf::num_instances)
        .def("get_best_split", [](const DomTreeLeaf& l) -> std::optional<py::tuple> {
            std::optional<DomTreeSplit> best_split = l.get_best_split();
            if (best_split)
                return encode_split(*best_split);
            return {};
        })
        .def("get_domains", &DomTreeLeaf::get_domains)
        .def("get_domain", &DomTreeLeaf::get_domain)
        .def("num_unreachable", &DomTreeLeaf::num_unreachable)
        .def("is_reachable", &DomTreeLeaf::is_reachable)
        .def("mark_unreachable", &DomTreeLeaf::mark_unreachable)
        .def("find_best_split", &DomTreeLeaf::find_best_split)
        .def("count_unreachable_leafs", &DomTreeLeaf::count_unreachable_leafs)
        .def("get_tree_bounds", &DomTreeLeaf::get_tree_bounds)
        .def("merge", &DomTreeLeaf::merge)
        .def(py::pickle(
            [](const DomTreeLeaf& l) -> py::bytes { // __getstate__
                std::ostringstream ss(std::ios::binary);
                l.to_binary(ss);
                return py::bytes(ss.str());
            },
            [](const py::bytes& bytes) -> DomTreeLeaf { // __setstate__
                std::istringstream ss(bytes);
                return DomTreeLeaf::from_binary(ss);
            }));




    py::class_<DomTreeT>(m, "RawDomTree")
        .def("root", [](const DomTreeT& t) { return t.root().id(); })
        .def("num_nodes", [](const DomTreeT& t) { return t.num_nodes(); })
        .def("is_root", [](const DomTreeT& t, NodeId n) { return t[n].is_root(); })
        .def("is_leaf", [](const DomTreeT& t, NodeId n) { return t[n].is_leaf(); })
        .def("is_internal", [](const DomTreeT& t, NodeId n) { return t[n].is_internal(); })
        .def("left", [](const DomTreeT& t, NodeId n) { return t[n].left().id(); })
        .def("right", [](const DomTreeT& t, NodeId n) { return t[n].right().id(); })
        .def("parent", [](const DomTreeT& t, NodeId n) { return t[n].parent().id(); })
        .def("depth", [](const DomTreeT& t, NodeId n) { return t[n].depth(); })
        .def("__str__", [](const DomTreeT& at) { return tostr(at); })
        .def("get_split", [](const DomTreeT& t, NodeId n) { return encode_split(t[n].get_split()); });
    */

        /*
    struct Optimizer {
        std::shared_ptr<FeatInfo> finfo;
        std::shared_ptr<DomainStore> store;
        std::shared_ptr<Solver> solver;
        std::shared_ptr<AddTree> at0;
        std::shared_ptr<AddTree> at1;
        std::shared_ptr<KPartiteGraph> g0;
        std::shared_ptr<KPartiteGraph> g1;
        std::shared_ptr<KPartiteGraphOptimize> opt;

        Optimizer()
            : finfo{}, solver{}, at0{}, at1{}, g0{}, g1{}, opt{}
        {
            at0 = std::make_shared<AddTree>(); // dummy addtree
            at1 = std::make_shared<AddTree>(); // dummy addtree
        }

        void init_graphs(std::unordered_set<FeatId> matches, bool match_is_reuse) {
            finfo = std::make_shared<FeatInfo>(*at0, *at1, matches, match_is_reuse);
            store = std::make_shared<DomainStore>();
            g0 = std::make_shared<KPartiteGraph>(&*store, *at0, *finfo, 0);
            g1 = std::make_shared<KPartiteGraph>(&*store, *at1, *finfo, 1);
        }

        void reset_opt() {
            opt = std::make_shared<KPartiteGraphOptimize>(&*store, *g0, *g1);
        }
    };

    py::class_<Optimizer>(m, "Optimizer")
        .def(py::init<>([](py::kwargs kwargs) -> Optimizer {
            Optimizer opt;

            if (kwargs.contains("minimize"))
                opt.at0 = std::make_shared<AddTree>(kwargs["minimize"].cast<AddTree>());
            if (kwargs.contains("maximize"))
                opt.at1 = std::make_shared<AddTree>(kwargs["maximize"].cast<AddTree>());

            std::unordered_set<FeatId> matches;
            bool match_is_reuse = true;
            if (kwargs.contains("matches"))
                matches = kwargs["matches"].cast<std::unordered_set<FeatId>>();
            if (kwargs.contains("match_is_reuse"))
                match_is_reuse = kwargs["match_is_reuse"].cast<bool>();

            opt.init_graphs(matches, match_is_reuse);

            //// simplify before generate opt, because opt sorts vertices!
            //// vertices need to be in DFS order
            //if (kwargs.contains("simplify") and py::isinstance<py::tuple>(kwargs["simplify"]))
            //{
            //    py::tuple t = kwargs["simplify"].cast<py::tuple>();
            //    FloatT max_err = t[0].cast<FloatT>();
            //    for (size_t i = 1; i < 5 && i < t.size();)
            //    {
            //        int instance = t[i++].cast<int>();
            //        bool overestimate = t[i++].cast<bool>();
            //        if (instance == 0)      opt.g0->simplify(max_err, overestimate);
            //        else if (instance == 1) opt.g1->simplify(max_err, overestimate);
            //    }
            //}

            opt.reset_opt();

            if (kwargs.contains("max_memory"))
                opt.store->set_max_mem_size(kwargs["max_memory"].cast<size_t>());

            return opt;
        }))
        .def("enable_smt", [](Optimizer& opt) {
            opt.solver = std::make_shared<Solver>(&*opt.finfo, *opt.at0, *opt.at1);
        })
        .def("set_smt_program", [](Optimizer& opt, const char *smt) {
            if (!opt.solver)
                throw std::runtime_error("smt not enabled");
            opt.solver->parse_smt(smt);
            //std::cout << opt.solver->get_z3() << std::endl;
        })
        .def("disable_smt", [](Optimizer& opt) {
            if (opt.solver)
            {
                //opt.solver->get_z3().reset();
                opt.solver.reset();
            }
        })
        .def("set_ara_eps", [](Optimizer& opt, FloatT eps, FloatT eps_incr) {
            opt.opt->set_eps(eps, eps_incr);
        })
        .def("use_dyn_prog_heuristic", [](Optimizer& opt) {
            opt.opt->use_dyn_prog_heuristic();
        })
        .def("get_ara_eps", [](const Optimizer& opt) {
            return opt.opt->get_eps();
        })
        .def("__str__", [](const Optimizer& opt) {
            std::stringstream ss;
            if (opt.solver)
                ss << "==== Z3 state: " << std::endl << opt.solver->get_z3() << std::endl;
            if (opt.g0)
                ss << std::endl << "==== KPartiteGraph 0 (minimized):" << std::endl
                   << *opt.g0 << std::endl;
            if (opt.g1)
                ss << std::endl << "==== KPartiteGraph 1 (maximized):" << std::endl
                   << *opt.g1 << std::endl;
            return ss.str();
        })
        .def("merge", [](Optimizer& opt, int K) {
            opt.g0->merge(K);
            opt.g1->merge(K);
            opt.reset_opt();
        })
        //.def("simplify", [](Optimizer opt, int instance, FloatT max_err, bool overestimate) {
        //    if (instance == 0) opt.g0->simplify(max_err, overestimate);
        //    else               opt.g1->simplify(max_err, overestimate);
        //    opt.reset_opt();
        //})
        .def("prune", [](Optimizer& opt) {
            if (!opt.solver)
                throw std::runtime_error("smt not enabled");

            auto f = [opt](const DomainBox& box) {
                z3::expr e = opt.solver->domains_to_z3(box);
                bool res = opt.solver->check(e);
                //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
                return res;
            };

            opt.g0->prune(f);
            opt.g1->prune(f);
            opt.reset_opt();
        })
        .def("prune", [](Optimizer& opt, const py::list& example, FloatT eps) {
            for (FeatId fid : opt.finfo->feat_ids0())
            {
                const py::handle& o = example[fid];
                if (py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o))
                {
                    FloatT v = o.cast<FloatT>();
                    auto f = [=](FeatId i) { return opt.finfo->get_id(0, i); };
                    opt.store->refine_workspace(LtSplit(fid, v-eps), false, f);
                    opt.store->refine_workspace(LtSplit(fid, v+eps), true, f);
                }
                //else throw std::runtime_error("not supported");
            }
            for (FeatId fid : opt.finfo->feat_ids1())
            {
                const py::handle& o = example[fid];
                if (py::isinstance<py::float_>(o))
                {
                    FloatT v = o.cast<FloatT>();
                    auto f = [=](FeatId i) { return opt.finfo->get_id(1, i); };
                    opt.store->refine_workspace(LtSplit(fid, v-eps), false, f);
                    opt.store->refine_workspace(LtSplit(fid, v+eps), true, f);
                }
                //else throw std::runtime_error("not supported");
            }

            DomainBox b = opt.store->get_workspace_box();
            auto f = [opt, &b](const DomainBox& box) {
                return box.overlaps(b);
            };

            opt.g0->prune(f);
            opt.g1->prune(f);
            opt.reset_opt();
            opt.store->clear_workspace();
        })
        .def("num_independent_sets", [](const Optimizer& opt, int instance) {
            if (instance == 0)
                return opt.g0->num_independent_sets();
            return opt.g1->num_independent_sets();
        })
        .def("num_vertices", [](const Optimizer& opt, int instance) {
            if (instance == 0)
                return opt.g0->num_vertices();
            return opt.g1->num_vertices();
        })
        .def("num_vertices", [](const Optimizer& opt, int instance, int indep_set) {
            if (instance == 0)
                return opt.g0->num_vertices_in_set(indep_set);
            return opt.g1->num_vertices_in_set(indep_set);
        })
        .def("get_used_feat_ids", [](Optimizer& opt) {
            return py::make_tuple(opt.finfo->feat_ids0(), opt.finfo->feat_ids1());
        })
        .def("xvar", [](Optimizer& opt, int instance, FeatId feat_id) {
            if (!opt.solver)
                throw std::runtime_error("smt not enabled");
            return opt.solver->xvar_name(instance, feat_id);
        })
        .def("xvar_id", [](const Optimizer& opt, int instance, FeatId feat_id) {
            return opt.finfo->get_id(instance, feat_id);
        })
        .def("num_solutions", [](const Optimizer& opt) {
            return opt.opt->solutions.size();
        })
        .def("solutions", [](const Optimizer& opt) {
            py::list l;
            for (auto& sol : opt.opt->solutions)
            {
                py::dict b;
                for (auto&& [id, d] : sol.box)
                    b[py::int_(id)] = d;
                l.append(py::make_tuple(sol.output0, sol.output1, b));
            }
            return l;
        })
        .def("epses", [](const Optimizer& opt) { return opt.opt->epses; })
        .def("nsteps", [](const Optimizer& opt) { return opt.opt->nsteps; })
        .def("nupdate_fails", [](const Optimizer& opt) { return opt.opt->nupdate_fails; })
        .def("nrejected", [](const Optimizer& opt) { return opt.opt->nrejected; })
        .def("nbox_filter_calls", [](const Optimizer& opt) { return opt.opt->nbox_filter_calls; })
        .def("current_bounds", [](const Optimizer& opt) { return opt.opt->current_bounds(); })
        .def("num_candidate_cliques", [](const Optimizer& opt) { return opt.opt->num_candidate_cliques(); })
        .def("memory", [](const Optimizer& opt) { return opt.store->get_mem_size(); })
        .def("step", [](Optimizer& opt, int nsteps, py::kwargs kwargs) {
            auto f = [opt](const DomainBox& box) {
                z3::expr e = opt.solver->domains_to_z3(box);
                bool res = opt.solver->check(e);
                //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
                //std::cout << opt.solver->get_z3() << std::endl;
                return res;
            };
            auto f_noz3 = [](const DomainBox&) { return true; };
            if (kwargs.contains("min_output_difference"))
            {
                FloatT min_output_difference = kwargs["min_output_difference"].cast<FloatT>();
                //std::cout << "step " << nsteps << " with min_output_difference " << min_output_difference << std::endl;
                if (opt.solver)
                    return opt.opt->steps(nsteps, f, min_output_difference);
                else
                    return opt.opt->steps(nsteps, f_noz3, min_output_difference);
            }
            else
            {
                FloatT max_output0 = std::numeric_limits<FloatT>::infinity();
                FloatT min_output1 = -std::numeric_limits<FloatT>::infinity();
                if (kwargs.contains("max_output"))
                    max_output0 = kwargs["max_output"].cast<FloatT>();
                if (kwargs.contains("min_output"))
                    min_output1 = kwargs["min_output"].cast<FloatT>();
                //std::cout << "step " << nsteps << " with max "
                //    << max_output0 << " and min "
                //    << min_output1 << std::endl;
                if (opt.solver)
                    return opt.opt->steps(nsteps, f, max_output0, min_output1);
                else
                    return opt.opt->steps(nsteps, f_noz3, max_output0, min_output1);
            }
        })
        .def("parallel", [](const Optimizer& opt, size_t nthreads) {
            return KPartiteGraphParOpt(nthreads, *opt.opt);
        })
        ;
        */

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
        .def("merge", &KPartiteGraph::merge)
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
        .def("prune_smt", [](KPartiteGraph& g, SMTSolver& solver) {
            g.prune([&solver](const DomainBox& box) {
                z3::expr e = solver.domains_to_z3(box);
                bool res = solver.check(e);
                //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
                return res;
            });
        })
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

    py::class_<KPartiteGraphOptimize>(m, "KPartiteGraphOptimize")
        .def(py::init<KPartiteGraph&, KPartiteGraph&>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("set_max_mem_size", [](KPartiteGraphOptimize& o, size_t m) { o.store().set_max_mem_size(m); })
        .def("get_max_mem_size", [](const KPartiteGraphOptimize& o) { return o.store().get_max_mem_size(); })
        .def("get_mem_size", [](const KPartiteGraphOptimize& o) { return o.store().get_mem_size(); })
        .def_readonly("num_steps", &KPartiteGraphOptimize::num_steps)
        .def_readonly("num_update_fails", &KPartiteGraphOptimize::num_update_fails)
        .def_readonly("num_rejected", &KPartiteGraphOptimize::num_rejected)
        .def_readonly("num_box_filter_calls", &KPartiteGraphOptimize::num_box_filter_calls)
        .def_readonly("solutions", &KPartiteGraphOptimize::solutions)
        .def_readonly("start_time", &KPartiteGraphOptimize::start_time)
        .def("num_solutions", [](const KPartiteGraphOptimize& o) { return o.solutions.size(); })
        .def("current_bounds", &KPartiteGraphOptimize::current_bounds)
        .def("num_candidate_cliques", &KPartiteGraphOptimize::num_candidate_cliques)
        .def("get_eps", &KPartiteGraphOptimize::get_eps)
        .def("set_eps", [](KPartiteGraphOptimize& o, FloatT eps) { o.set_eps(eps); })
        .def("use_dyn_prog_heuristic", &KPartiteGraphOptimize::use_dyn_prog_heuristic)
        //.def("__str__", [](const KPartiteGraphOptimize& o) { return tostr(o); })
        .def("steps", [](KPartiteGraphOptimize& opt, int nsteps, py::kwargs kwargs) {
            //auto f = [opt](const DomainBox& box) {
            //    z3::expr e = opt.solver->domains_to_z3(box);
            //    bool res = opt.solver->check(e);
            //    //std::cout << "test: " << box << " -> " << e << " res? " << res << std::endl;
            //    //std::cout << opt.solver->get_z3() << std::endl;
            //    return res;
            //};
            auto f_noz3 = [](const DomainBox&) { return true; };
            if (kwargs.contains("min_output_difference"))
            {
                FloatT min_output_difference = kwargs["min_output_difference"].cast<FloatT>();
                //if (opt.solver)
                //    return opt.opt->steps(nsteps, f, min_output_difference);
                return opt.steps(nsteps, f_noz3, min_output_difference);
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
                return opt.steps(nsteps, f_noz3, max_output0, min_output1);
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



} /* PYBIND11_MODULE */
