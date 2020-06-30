/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * ----
 *
 * This file contains reimplemplementations of concepts introduced by the
 * following paper Chen et al. 2019:
 *
 * https://papers.nips.cc/paper/9399-robustness-verification-of-tree-based-models
 * https://github.com/chenhongge/treeVerification
 */

#ifndef TREECK_GRAPH_H
#define TREECK_GRAPH_H

#include <tuple>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#include "domain.h"
#include "tree.h"

namespace treeck {

    // to avoid having to check whether a domain is real or bool, we use RealDomain for both
    // True = [0.0, 1.0), False = [1.0, 2.0), Everything = [0.0, 2.0)
    // (like a LtSplit(_, 1.0))
    using DomainT = RealDomain;
    const DomainT BOOL_DOMAIN{static_cast<FloatT>(0.0), static_cast<FloatT>(2.0)};
    const DomainT TRUE_DOMAIN{static_cast<FloatT>(0.0), static_cast<FloatT>(1.0)};
    const DomainT FALSE_DOMAIN{static_cast<FloatT>(1.0), static_cast<FloatT>(2.0)};
    using DomainPair = std::pair<int, DomainT>;


    class DomainBox;

    using BoxFilter = const std::function<bool(const DomainBox&)>&;
    using FeatIdMapper = const std::function<int(FeatId)>&;


    class FeatInfo {
    public:
        const int UNUSED_ID = -1;

    private:
        std::vector<FeatId> feat_ids0_; // feat ids used in at0
        std::vector<FeatId> feat_ids1_; // feat ids used in at1

        std::unordered_map<int, int> key2id_;
        std::vector<bool> is_real_;
        int max_id_;
        int id_boundary_;

    public:
        FeatInfo();
        //FeatInfo(const AddTree& at0);
        FeatInfo(const AddTree& at0,
                 const AddTree& at1,
                 const std::unordered_set<FeatId>& matches,
                 bool match_is_reuse);

        int get_max_id() const;
        size_t num_ids() const;
        int get_id(int instance, FeatId feat_id) const;
        bool is_instance0_id(int id) const;

        bool is_real(int id) const;

        const std::vector<FeatId>& feat_ids0() const;
        const std::vector<FeatId>& feat_ids1() const;
    };



    class DomainStore {
        using Block = std::vector<DomainPair>;

        std::vector<Block> store_;
        Block workspace_;
        size_t max_mem_size_;

        Block& get_block_with_capacity(size_t cap);

    public:
        size_t get_mem_size() const;
        void set_max_mem_size(size_t max_mem);

        DomainStore();

        /** get the workspace DomainBox. (!) don't store somewhere, pointers in DomainBox not stable */
        DomainBox get_workspace_box() const;
        DomainBox push_workspace();

        void refine_workspace(Split split, bool is_left_child, FeatIdMapper fmap);
        void combine_in_workspace(const DomainBox& a, const DomainBox& b);
        DomainBox combine_and_push(const DomainBox& a, const DomainBox& b);
        void clear_workspace();
    };

    class DomainBox {
    public:
        using const_iterator = const DomainPair *;

    private:
        const_iterator begin_;
        const_iterator end_;

    public:
        DomainBox(const DomainPair *begin, const DomainPair *end);
        static DomainBox null_box();

        inline const DomainPair *begin() const { return begin_; }
        inline const DomainPair *end() const { return end_; }

        bool overlaps(const DomainBox& other) const;
        size_t size() const;

        //bool is_right_neighbor(const DomainBox& other) const;
        //void join_right_neighbor(const DomainBox& other);
    };

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box);




    struct Vertex {
        DomainBox box;
        FloatT output;
        FloatT min_bound;
        FloatT max_bound;

        Vertex(DomainBox box, FloatT output);

        //bool operator<(const Vertex& other) const;
        //bool operator>(const Vertex& other) const;
    };




    struct IndependentSet {
        std::vector<Vertex> vertices;
    };




    class KPartiteGraph {
        DomainStore *store_;
        std::vector<IndependentSet> sets_;
        friend class KPartiteGraphOptimize;

    private:
        void fill_independence_set(IndependentSet& set,
                AddTree::TreeT::CRef node,
                FeatIdMapper fmap);

    public:
        KPartiteGraph(DomainStore *store_);
        //KPartiteGraph(DomainStore *store_, const AddTree& addtree);
        KPartiteGraph(DomainStore *store_, const AddTree& addtree, FeatIdMapper fmap);
        KPartiteGraph(DomainStore *store_, const AddTree& addtree, const FeatInfo& finfo, int instance);

        std::vector<IndependentSet>::const_iterator begin() const;
        std::vector<IndependentSet>::const_iterator end() const;

        /** remove all vertices for which the given function returns true. */
        void prune(BoxFilter filter);

        std::tuple<FloatT, FloatT> propagate_outputs();
        void merge(int K);
        void simplify(FloatT max_err, bool overestimate); // vertices must be in DFS order, left to right!
        void sort_asc();
        void sort_desc();
        void sort_bound_asc();
        void sort_bound_desc();

        size_t num_independent_sets() const;
        size_t num_vertices() const;
        size_t num_vertices_in_set(int indep_set) const;
    };

    std::ostream& operator<<(std::ostream& s, const KPartiteGraph& graph);




    template <typename T>
    using two_of = std::tuple<T, T>;

    struct CliqueInstance {
        FloatT output;              // A*'s g(clique_instance)
        FloatT heuristic;          // A*'s h(clique_instance)

        inline FloatT output_bound(FloatT eps = 1.0) const
        {
            return output + eps * heuristic;
        } // g(clique_instance) + eps * h(clique_instance)

        short indep_set; // index of tree (= independent set in graph) to merge with
        int vertex;      // index of next vertex to merge from `indep_set` (must be a compatible one!)
    };

    struct Clique {
        DomainBox box;

        two_of<CliqueInstance> instance;
    };

    struct CliqueMaxDiffPqCmp {
        FloatT eps;
        bool operator()(const Clique&, const Clique&) const;
    };

    std::ostream& operator<<(std::ostream& s, const CliqueInstance& ci);
    std::ostream& operator<<(std::ostream& s, const Clique& c);



    struct Solution {
        DomainBox box;
        FloatT output0, output1;
    };

    std::ostream& operator<<(std::ostream& s, const Solution& sol);


    class KPartiteGraphOptimize {
        DomainStore *store_;
        two_of<const KPartiteGraph&> graph_; // <0> minimize, <1> maximize

        // a vector ordered as a pq containing "partial" cliques (no max-cliques)
        std::vector<Clique> cliques_;
        CliqueMaxDiffPqCmp cmp_;
        FloatT eps_incr_;

        enum { DYN_PROG, RECOMPUTE } heuristic_type;

    private:
        Clique pq_pop();
        void pq_push(Clique&& c);

        bool is_solution(const Clique& c) const;

        template <size_t instance>
        bool is_instance_solution(const Clique& c) const;

        template <size_t instance>
        bool update_clique(Clique& c);

        template <size_t instance, typename BF, typename OF>
        void step_instance(Clique c, BF box_filter, OF output_filter);

        template <size_t instance, typename BF, typename OF>
        void expand_clique_instance(Clique c, BF box_filter, OF output_filter);

        template <typename BF, typename OF>
        bool step_aux(BF bf, OF of);

    public:
        two_of<size_t> nsteps;
        size_t nupdate_fails;
        size_t nrejected;
        size_t nbox_filter_calls;

        std::vector<Solution> solutions;
        std::vector<FloatT> epses;

    public:
        //KPartiteGraphOptimize(KPartiteGraph& g0); // minimize g0
        //KPartiteGraphOptimize(bool maximize, KPartiteGraph& g1); // maximize g1
        KPartiteGraphOptimize(KPartiteGraph& g0, KPartiteGraph& g1); // minimize g0, maximize g1
        KPartiteGraphOptimize(DomainStore *store, KPartiteGraph& g0, KPartiteGraph& g1);

        FloatT get_eps() const;
        void set_eps(FloatT eps, FloatT eps_incr);
        void use_dyn_prog_heuristic();

        bool step();
        bool step(BoxFilter bf);
        bool step(BoxFilter bf, FloatT max_output0, FloatT min_output1);
        bool step(BoxFilter bf, FloatT min_output_difference);
        bool steps(int howmany);
        bool steps(int howmany, BoxFilter bf);
        bool steps(int howmany, BoxFilter bf, FloatT max_output0, FloatT min_output1);
        bool steps(int howmany, BoxFilter bf, FloatT min_output_difference);

        two_of<FloatT> current_bounds() const;
        size_t num_candidate_cliques() const;
    };

} /* namespace treeck */

#endif /* TREECK_GRAPH_H */
