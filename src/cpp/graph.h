/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_H
#define VERITAS_GRAPH_H

#include <tuple>
#include <vector>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "domain.h"
#include "tree.h"

namespace veritas {

    // to avoid having to check whether a domain is real or bool, we use RealDomain for both
    // True = [0.0, 1.0), False = [1.0, 2.0), Everything = [0.0, 2.0)
    // (like a LtSplit(_, 1.0))
    using DomainT = RealDomain;
    const DomainT BOOL_DOMAIN;
    const DomainT FALSE_DOMAIN = DomainT::from_hi_exclusive(1.0);
    const DomainT TRUE_DOMAIN = DomainT::from_lo(1.0);
    using DomainPair = std::pair<int, DomainT>;


    class DomainBox;
    class DomainStore;

    using GenericBoxFilter = const std::function<bool(const DomainBox&)>&;
    using GenericBoxFilterT = std::remove_const_t<std::remove_reference_t<GenericBoxFilter>>;
    using GenericBoxChecker = const std::function<bool(DomainStore&)>&;
    using GenericBoxCheckerT = std::remove_const_t<std::remove_reference_t<GenericBoxChecker>>;
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
        size_t get_used_mem_size() const;
        size_t get_max_mem_size() const;
        void set_max_mem_size(size_t max_mem);

        DomainStore();

        inline std::vector<DomainPair>& workspace() { return workspace_; };

        /** get the workspace DomainBox. (!) don't store somewhere, pointers in DomainBox not stable */
        DomainBox get_workspace_box() const;
        DomainBox push_workspace();

        void refine_workspace(Split split, bool is_left_child, FeatIdMapper fmap);
        void combine_in_workspace(const DomainBox& a, const DomainBox& b, bool copy_b=true);
        DomainBox combine_and_push(const DomainBox& a, const DomainBox& b, bool copy_b=true);
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
        DomainStore store_;
        std::vector<IndependentSet> sets_;
        friend class KPartiteGraphOptimize;

    private:
        void fill_independence_set(IndependentSet& set,
                AddTree::TreeT::CRef node,
                FeatIdMapper fmap);

    public:
        KPartiteGraph();
        //KPartiteGraph(const AddTree& addtree);
        KPartiteGraph(const AddTree& addtree, FeatIdMapper fmap);
        KPartiteGraph(const AddTree& addtree, const FeatInfo& finfo, int instance);

        std::vector<IndependentSet>::const_iterator begin() const;
        std::vector<IndependentSet>::const_iterator end() const;

        /** remove all vertices for which the given function returns true. */
        void prune(GenericBoxFilter filter);

        /** remove all vertices that don't overlap with box in workspace.
            and crop vertices' boxes so they are contained by the box in the workspace.
            clear the workspace */
        void prune_by_workspace_box();

        std::tuple<FloatT, FloatT> propagate_outputs();
        std::tuple<FloatT, FloatT> basic_bound() const;
        void merge(int K);
        //void simplify(FloatT max_err, bool overestimate); // vertices must be in DFS order, left to right!
        void sort_asc();
        void sort_desc();
        void sort_bound_asc();
        void sort_bound_desc();

        size_t num_independent_sets() const;
        size_t num_vertices() const;
        size_t num_vertices_in_set(int indep_set) const;

        inline const DomainStore& store() const { return store_; }
        inline DomainStore& store() { return store_; }

        // for merge: with two ensembles, they simply merge independent sets of (-ensemble1 + ensemble2)
        void add_with_negated_leaf_values(const KPartiteGraph&);
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

        inline FloatT output_difference(FloatT eps = 1.0) const {
            return std::get<1>(instance).output_bound(eps)
                - std::get<0>(instance).output_bound(eps);
        }
    };

    struct CliqueMaxDiffPqCmp {
        FloatT eps;
        inline bool operator()(const Clique& a, const Clique& b) const {
            return a.output_difference(eps) < b.output_difference(eps);
        }
    };

    std::ostream& operator<<(std::ostream& s, const CliqueInstance& ci);
    std::ostream& operator<<(std::ostream& s, const Clique& c);



    struct Solution {
        DomainBox box;
        FloatT output0, output1;
        FloatT eps;
        double time;
        bool is_valid;

        inline FloatT output_difference() const { return output1 - output0; }
    };

    std::ostream& operator<<(std::ostream& s, const Solution& sol);


    class KPartiteGraphOptimize {
        friend class KPartiteGraphParOpt;

        DomainStore store_;
        two_of<const KPartiteGraph&> graph_; // <0> minimize, <1> maximize

        // a vector ordered as a pq containing "partial" cliques (no max-cliques)
        std::vector<Clique> cliques_;
        FloatT last_bound_;
        CliqueMaxDiffPqCmp cmp_;

    public:
        enum Heuristic { DYN_PROG, RECOMPUTE };

    private:
        Heuristic heuristic_;

    private:
        Clique pq_pop();
        void pq_push(Clique&& c);

        bool is_solution(const Clique& c) const;

        template <size_t instance>
        bool is_instance_solution(const Clique& c) const;

        template <size_t instance>
        bool update_clique(Clique& c);

        template <size_t instance, typename BC, typename OF>
        void step_instance(Clique&& c, BC box_checker, OF output_filter);

        template <size_t instance, typename BC, typename OF>
        void expand_clique_instance(Clique&& c, BC box_checker, OF output_filter);

        template <typename BC, typename OF>
        bool step_aux(BC box_checker, OF of);

    public:
        two_of<size_t> num_steps;
        size_t num_update_fails;
        size_t num_rejected;
        size_t num_box_checks;

        std::vector<Solution> solutions;
        double start_time;

    public:
        KPartiteGraphOptimize(KPartiteGraph& g0, KPartiteGraph& g1, Heuristic heur = RECOMPUTE);

        /** copy states i, i+K, i+2K,... from `other` */
        KPartiteGraphOptimize(const KPartiteGraphOptimize& other, size_t i, size_t K);

        FloatT get_eps() const;
        void set_eps(FloatT eps, bool rebuild_heap = true);

        bool step();
        bool step(GenericBoxChecker bc);
        bool step(GenericBoxChecker bc, FloatT max_output0, FloatT min_output1);
        bool step(GenericBoxChecker bc, FloatT min_output_difference);
        bool steps(int howmany);
        bool steps(int howmany, GenericBoxChecker bc);
        bool steps(int howmany, GenericBoxChecker bc, FloatT max_output0, FloatT min_output1);
        bool steps(int howmany, GenericBoxChecker bc, FloatT min_output_difference);

        two_of<FloatT> current_bounds() const;
        size_t num_candidate_cliques() const;

        const KPartiteGraph& graph0() const;
        const KPartiteGraph& graph1() const;

        inline const DomainStore& store() const { return store_; }
        inline DomainStore& store() { return store_; }
    };

    class Worker {
        friend class KPartiteGraphParOpt;

        size_t index_;

        bool work_flag_;
        bool stop_flag_; // false to disable task
        enum { RDIST_DISABLED, RDIST_SETUP, RDIST_READY, RDIST_GO, RDIST_DONE,
            RDIST_STORE } redistribute_;
        size_t num_millisecs_; // 0 to disable task

        size_t new_valid_solutions_; // new valid solutions since last `steps_for` call

        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::optional<KPartiteGraphOptimize> opt_;
        GenericBoxCheckerT box_checker_;

    public:
        Worker();
    };

    struct SharedWorkerInfo {
        FloatT max_output0;
        FloatT min_output1;
        FloatT min_output_difference;
        FloatT best_bound;
        FloatT new_eps;
        SharedWorkerInfo(FloatT eps);
    };

    class KPartiteGraphParOpt {
        std::unique_ptr<std::deque<Worker>> workers_;
        std::unique_ptr<SharedWorkerInfo> info_;

        void wait();
        void redistribute_work();
        static void worker_fun(std::deque<Worker> *workers, const SharedWorkerInfo*, size_t self_index);

    public:
        KPartiteGraphParOpt(size_t num_threads,
                const KPartiteGraphOptimize& opt);

        void join_all();

        void steps_for(size_t num_millisecs);
        FloatT get_eps() const;
        void set_eps(FloatT new_eps);

        inline size_t num_threads() const { return workers_->size(); }
        const KPartiteGraphOptimize& worker_opt(size_t worker) const;

        template <typename F>
        inline void set_box_checker(F f)
        {
            for (size_t i = 0; i < num_threads(); ++i)
            {
                Worker& w = workers_->at(i);
                std::lock_guard guard(w.mutex_);
                w.box_checker_ = f();
            }
        }
        void set_output_limits(FloatT max_output0, FloatT min_output1);
        void set_output_limits(FloatT min_output_difference);

        size_t num_solutions() const;
        size_t num_new_valid_solutions() const; // valid solutions since last 'steps_for' call
        size_t num_candidate_cliques() const;
        two_of<FloatT> current_bounds() const;
        std::vector<size_t> current_memory() const;
    };

} /* namespace veritas */

#endif /* VERITAS_GRAPH_H */
