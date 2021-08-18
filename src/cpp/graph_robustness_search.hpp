/**
 * \file graph_robustness_search.hpp
 *
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP
#define VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP

#include "domain.hpp"
#include "tree.hpp"
#include "graph.hpp"
#include "constraints.hpp"
#include <iostream>
#include <chrono>
#include <map>

namespace veritas {

    /** \private */
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    /** \private */
    struct State {
        FloatT g, h;
        BoxRef box;

        int indep_set;
        bool is_expanded;
    };

    /** \private */
    struct SolutionRef {
        size_t state_index;
        FloatT eps;
        double time;
    };

    /** A collection of statistics of the GraphSearch at specific points in
     * time. After GraphSearch::steps() finishes, a snapshot is added to
     * GraphSearch::snapshots. */
    struct Snapshot {
        double time = 0.0;
        size_t num_steps = 0;
        size_t num_solutions = 0;
        size_t num_states = 0;
        FloatT eps = 0.0;
        std::tuple<FloatT, FloatT, FloatT> bounds = {-FLOATT_INF, FLOATT_INF, FLOATT_INF}; // lo, up_a, up_ara
    };

    class GraphOutputSearch;
    class GraphRobustnessSearch;

    struct OutputCmp {
        const GraphOutputSearch& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;
        inline FloatT fscore(const State& s) const { return s.g + eps*s.h; }
        inline bool operator()(const State& a, const State& b) const
        { return fscore(a) < fscore(b); }
    };

    struct RobustnessCmp {
        const GraphRobustnessSearch& search;

        bool operator()(size_t i, size_t j) const;
        inline bool operator()(const State& a, const State& b) const
        { return false; /* TODO */ }
    };


    /**
     * Subclass this with `Curiously recurring template pattern`, and heap
     * comparison type to sort solutions.
     *
     * Methods to implement:
     *  - Derived::push_state
     *      * call GraphSearch::push_state_, which returns `state_index`..
     *      * .. then add this state to heap(s)
     *  - Derived::push_solution
     *      * call GraphSearch::push_solution_, which returns `solution_index`..
     *      * .. then (if applicable) check for solution optimality
     *  - Derived::step
     *      * call GraphSearch::step_, which takes a valid popped-from-heap
     *      `state_index`, and returns whether there are more
     *      * should return true when all relevant states have been explored,
     *      false otherwise.
     *  - Derived::steps
     *      * call GraphSearch::steps_ ..
     *      * .. then, if applicable, push a snapshot
     *  - Derived::expand
     *      * call GraphSearch::expand_ ..
     *      * .. then eg. update eps
     *
     *  - Derived::compute_score(State& state)
     *      * fills in g and h of given state
     */
    template <typename Derived, typename CmpT>
    class GraphSearch {
    protected:
        AddTree at_;
        Graph g_;
        
        BlockStore<DomainPair> store_;
        mutable struct {
            Box box;
        } workspace_;

        std::vector<State> states_;
        std::vector<SolutionRef> solutions_; // indices into states_
        time_point start_time_;
        size_t num_steps_;
        CmpT solution_cmp_;

        friend OutputCmp;
        friend RobustnessCmp;

        GraphSearch(const AddTree& at, const CmpT& solution_cmp)
            : at_(at.neutralize_negative_leaf_values())
            , g_(at_)
            , start_time_{std::chrono::system_clock::now()}
            , num_steps_{0}
            , solution_cmp_(solution_cmp)
        {}

        /** call in Derived constructor */
        void init()
        {
            State state = {
                0.0, 0.0, // g, h
                BoxRef::null_box(), // box
                -1, // indep_set
                false, // is_expanded
            };
            Graph::Vertex fake_v{0, BoxRef::null_box(), 0.0};
            State fake_parent = state; // copy
            static_cast<Derived *>(this)->compute_score(state, fake_parent, fake_v);
            static_cast<Derived *>(this)->push_state(std::move(state));
        }

    public:
        void prune_by_box(const BoxRef& box)
        {
            if (states_.size() > 1)
                throw std::runtime_error("invalid state: pruning with more than 1 state");
            g_.prune_by_box(box, false);
        }

    protected:
        bool is_solution(const State& state) const
        {
            return state.indep_set + 1 == static_cast<int>(g_.num_independent_sets());
        }

    protected:
        /** Override. Call this from Derived::push_state(State&&).
         * \return State index */
        size_t push_state_(State&& state)
        {
            size_t state_index = states_.size();
            states_.push_back(std::move(state));
            return state_index;
        }

        /** Override. Call this from Derived::push_solution(size_t).
         * \return Solution index */
        size_t push_solution_(size_t state_index)
        {
            states_[state_index].is_expanded = true;
            solutions_.push_back({ state_index, 0.0, time_since_start() });

            // sort solutions
            size_t i = solutions_.size()-1;
            for (; i > 0; --i)
            {
                SolutionRef& sol1 = solutions_[i-1];
                SolutionRef& sol2 = solutions_[i];
                if (solution_cmp_(states_[sol1.state_index], states_[sol2.state_index]))
                    std::swap(sol1, sol2);
                else return i;
            }
            return 0;
        }

        /** Override. Call this from Derived::step(). */
        void step_(size_t state_index)
        {
            // this state is a solution
            if (is_solution(states_[state_index]))
                static_cast<Derived *>(this)->push_solution(state_index);
            else
                static_cast<Derived *>(this)->expand(state_index);

            ++num_steps_;
        }

        /** Override. Call this from Derived::steps(). */
        bool steps_(size_t num_steps)
        {
            bool done = false;
            size_t num_sol = num_solutions();
            for (size_t i = 0; i < num_steps && num_sol == num_solutions(); ++i)
            {
                if (static_cast<Derived *>(this)->step())
                {
                    done = true;
                    break;
                }
            }
            return done;
        }

        /** Override. Call this from Derived::expand(size_t). */
        void expand_(size_t state_index)
        {
            states_[state_index].is_expanded = true;
            BoxRef parent_box = states_[state_index].box;

            Graph::IndepSet set = g_.get_vertices(states_[state_index].indep_set + 1);
            int num_vertices = static_cast<int>(set.size());
            for (int vertex = 0; vertex < num_vertices; ++vertex)
            {
                const Graph::Vertex& v = set[vertex];
                if (v.box.overlaps(parent_box))
                {
                    combine_boxes(v.box, parent_box, true, workspace_.box);
                    construct_and_push_states(state_index, v);
                }
            }
        }

    public:
        bool step_for(double num_seconds, size_t num_steps)
        {
            double start = time_since_start();
            bool done = false;

            while (!done)
            {
                if (static_cast<Derived *>(this)->stop_conditions_met())
                    break;

                double dur = time_since_start() - start;
                done = static_cast<Derived *>(this)->steps(num_steps);
                if (dur >= num_seconds)
                    break;
            }

            return done;
        }

        size_t num_states() const { return states_.size(); }
        size_t num_solutions() const { return solutions_.size(); }
        size_t remaining_mem_capacity() const
        { return (size_t(1024)*1024*1024) - store_.get_mem_size(); }

        /** Seconds since the construction of the search */
        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time_).count() * 1e-6;
        }

    private:

        void construct_and_push_states(size_t parent_state_index, const Graph::Vertex& v)
        {
            auto push_workspace_box_fun = [this, parent_state_index, &v](Box& b){
                BoxRef box = BoxRef(store_.store(b, remaining_mem_capacity()));
                State new_state = {
                    0.0,
                    0.0, // heuristic, set later, after `in_visited` check, in `compute_heuristic`
                    box,
                    -1, // indep_sets, set later with h in `compute_heuristic`
                    false, // is_expanded
                };

                static_cast<Derived *>(this)
                    ->compute_score(new_state, states_[parent_state_index], v);

                if (!std::isinf(new_state.h))
                    static_cast<Derived *>(this)->push_state(std::move(new_state));
                else std::cout << " -> inf h, skipping" << std::endl;
            };

            //if (constr_prop)
            //{
            //    constr_prop->check(workspace_.box, push_workspace_box_fun);
            //    //constr_prop->print();
            //}
            //else
            //{
                push_workspace_box_fun(workspace_.box);
            //}

            workspace_.box.clear();
        }

    protected:
        FloatT compute_output_heuristic(const State& state) const
        {
            // compute heuristic
            FloatT h = 0.0;
            for (size_t indep_set = state.indep_set; indep_set < g_.num_independent_sets(); ++indep_set)
            {
                FloatT max = -FLOATT_INF;
                Graph::IndepSet set = g_.get_vertices(indep_set);
                for (const auto& v : g_.get_vertices(indep_set))
                    if (v.box.overlaps(state.box))
                        max = std::max(max, v.output);

                h += max;
            }

            return h;
        }
        
        FloatT compute_output_heuristic_dynprog(const State& state) const
        {
            FloatT h = 0.0;
            std::vector<FloatT> d0, d1;
            int prev_indep_set = -1;
            for (size_t indep_set = state.indep_set; indep_set < g_.num_independent_sets(); ++indep_set)
            {
                FloatT tree_max = -FLOATT_INF;
                //
                // fill d0 with the first indep_set
                if (d0.empty())
                {
                    for (const auto& v : g_.get_vertices(indep_set))
                    {
                        if (v.box.overlaps(state.box))
                        {
                            d0.push_back(v.output);
                            tree_max = std::max(tree_max, v.output);
                        }
                        else
                        {
                            d0.push_back(-FLOATT_INF);
                        }
                    }
                }
                else
                {
                    const auto& set0 = g_.get_vertices(prev_indep_set);
                    const auto& set1 = g_.get_vertices(indep_set);
                    for (const auto& v1 : set1)
                    {
                        FloatT max = -FLOATT_INF;

                        if (v1.box.overlaps(state.box))
                        {
                            tree_max = std::max(tree_max, v1.output);
                            for (size_t j = 0; j < d0.size(); ++j)
                            {
                                const auto& v0 = set0[j];
                                if (v0.box.overlaps(v1.box) && max < d0[j])
                                    max = d0[j];
                            }
                            d1.push_back(max + v1.output);
                        }
                        else
                        {
                            d1.push_back(-FLOATT_INF);
                        }
                    }

                    d0.clear();
                    std::swap(d0, d1);
                }

                prev_indep_set = indep_set;
            }
            return h;
        }
    }; // class GraphSearch

    class GraphOutputSearch : public GraphSearch<GraphOutputSearch, OutputCmp> {
        std::vector<size_t> a_heap_, ara_heap_; // indexes into states_
        OutputCmp a_cmp_, ara_cmp_;
        FloatT eps_increment_;
        double last_eps_increment_, avg_eps_update_time_;

        using CmpT = OutputCmp;

        std::vector<Snapshot> snapshots_;

    public:
        friend GraphSearch<GraphOutputSearch, OutputCmp>;

        GraphOutputSearch(const AddTree& at)
            : GraphSearch(at, a_cmp_)
            , a_cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.01}
            , eps_increment_{0.01}
            , last_eps_increment_{0.0}
            , avg_eps_update_time_{0.0}
        {
            init();
            push_snapshot();
        }

    private:
        void compute_score(State& new_state, const State& parent_state,
                const Graph::Vertex& merged_vertex) const
        {
            new_state.g = parent_state.g + merged_vertex.output;
            new_state.h = compute_output_heuristic(new_state);
        }

        void push_state(State&& state)
        {
            size_t state_index = push_state_(std::move(state));
            push_to_heap(a_heap_, state_index, a_cmp_);
            if (ara_cmp_.eps < 1.0)
                push_to_heap(ara_heap_, state_index, ara_cmp_);
        }

        void push_solution(size_t state_index)
        {
            size_t solution_index = push_solution_(state_index);
            SolutionRef& sol = solutions_[solution_index];

            // if the solution state has a better score than the top of the A*
            // heap, then this is an optimal solution
            if (a_heap_.size() == 0 
                    or ara_heap_.size() == 0
                    or a_cmp_.fscore(states_[state_index])
                    >= a_cmp_.fscore(states_[a_heap_.front()]))
            {
                sol.eps = 1.0;
                set_eps(1.0); // disabled ARA* by emptying ara_heap_
            }
            else
            {
                sol.eps = ara_cmp_.eps;
                update_eps(eps_increment_);
            }

            // better solutions have an eps at least as good as this one
            while (solution_index != 0)
            {
                --solution_index;
                FloatT& eps = solutions_[solution_index].eps;
                eps = std::max(sol.eps, eps);
            }
        }

        void expand(size_t state_index)
        {
            expand_(state_index);

            // if the previous best suboptimal solution was still in the ara
            // stack, would it be at the top? if so, increase eps
            if (num_solutions() > 0 && ara_heap_.size() > 0)
            {
                const State& s0 = states_[state_index];
                for (SolutionRef& sol : solutions_)
                {
                    const State& s1 = states_[sol.state_index];
                    if (sol.eps != 1.0 && ara_cmp_(s0, s1))
                    {
                        std::cout << "UPDATE PREVIOUS SOLUTION " << ara_cmp_.fscore(s1)
                            << " > " << ara_cmp_.fscore(s0)
                            << " (" << ara_cmp_.eps << " ==? " << ara_cmp_.eps << ")";
                        sol.eps = ara_cmp_.eps;
                        update_eps(eps_increment_);
                        //std::cout << " new eps=" << ara_cmp_.eps << std::endl;
                    }
                    else break;
                }
            }

            // if finding another suboptimal solution takes too long,
            // decrease eps, halve eps_increment_
            double t = time_since_start();
            if (last_eps_increment_ > 0.0 && (t-last_eps_increment_) > 20*avg_eps_update_time_)
            {
                std::cout << "HALVING eps_increment_ to " << eps_increment_/2.0
                    << " avg t: " << avg_eps_update_time_
                    << " t: " << (t-last_eps_increment_);
                eps_increment_ = std::max(0.001, eps_increment_/2.0);
                update_eps(-eps_increment_);
                std::cout
                    << " (eps=" << ara_cmp_.eps << ")"
                    << std::endl;
            }
        }
        
    public:
        bool step()
        {
            size_t state_index;
            while (true)
            {
                if (a_heap_.empty())
                    return true; // we're done
                state_index = pop_state();
                if (!states_[state_index].is_expanded)
                    break;
            }
            step_(state_index);
            return false;
        }

        bool steps(size_t num_steps)
        {
            bool done = steps_(num_steps);
            push_snapshot();
            return done;
        }

    private:
        void push_to_heap(std::vector<size_t>& heap, size_t state_index, const OutputCmp& cmp)
        {
            heap.push_back(state_index);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }

        size_t pop_state()
        {
            size_t state_index;
            if (!ara_heap_.empty() && num_steps_%2 == 1)
                state_index = pop_from_heap(ara_heap_, ara_cmp_);
            else
                state_index = pop_from_heap(a_heap_, a_cmp_);
            return state_index;
        }

        size_t pop_from_heap(std::vector<size_t>& heap, const OutputCmp& cmp)
        {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            size_t state_index = heap.back();
            heap.pop_back();
            return state_index;
        }

        void push_snapshot()
        {
            snapshots_.push_back({
                time_since_start(),
                num_steps_,
                num_solutions(),
                num_states(),
                num_solutions() > 0 ? solutions_[0].eps : FloatT(0.0),
                current_bounds_with_base_score()});
        }

        /** ARA* lower, A* upper, ARA* upper */
        std::tuple<FloatT, FloatT, FloatT> current_bounds() const
        {
            FloatT lo = -FLOATT_INF, up_ara = FLOATT_INF;
            if (num_solutions() > 0)
            {
                const SolutionRef& s = solutions_[0];
                lo = states_[s.state_index].g; // best solution so far, sols are sorted
                up_ara = lo / s.eps;
            }
            FloatT up_a = a_cmp_.fscore(states_[a_heap_.front()]);
            return {lo, up_a, up_ara};
        }

        void set_eps(FloatT eps)
        {
            ara_cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps));
            if (ara_cmp_.eps < 1.0)
                std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            else
                ara_heap_.clear();
        }

    public:
        /** ARA* lower, A* upper, ARA* upper */
        std::tuple<FloatT, FloatT, FloatT> current_bounds_with_base_score() const
        {
            auto &&[lo, up_a, up_ara] = current_bounds();
            return {lo+at_.base_score, up_a+at_.base_score, up_ara+at_.base_score};
        }

        void update_eps(FloatT added_value) // this updates time, set_eps does not
        {
            set_eps(ara_cmp_.eps + added_value);

            double t = time_since_start();
            double time_since_previous_incr = t - last_eps_increment_;
            //if (time_since_previous_incr*10 < avg_eps_update_time_)
            //{
            //    eps_increment_ *= 2;
            //    std::cout << "DOUBLING eps_increment_ to " << eps_increment_
            //        << " avg t: " << avg_eps_update_time_
            //        << " t: " << (t - last_eps_increment_)
            //        << std::endl;
            //}
            last_eps_increment_ = t;
            avg_eps_update_time_ = 0.2*avg_eps_update_time_ + 0.8*time_since_previous_incr;
        }



    };

    //class GraphRobustnessSearch : public GraphSearch<GraphRobustnessSearch, RobustnessCmp> {

    //    std::vector<size_t> heap_; // indexes into states_
    //    RobustnessCmp cmp_;

    //    std::vector<FloatT> example_;
    //    FloatT max_delta_;

    //    using CmpT = RobustnessCmp;

    //public:
    //    friend GraphSearch<GraphRobustnessSearch, RobustnessCmp>;

    //    GraphRobustnessSearch(const AddTree& at, const std::vector<FloatT>& example, FloatT max_delta)
    //        : GraphSearch(at, cmp_)
    //        , cmp_{*this}
    //        , example_(example)
    //        , max_delta_(max_delta)
    //    {

    //    }

    //private:
    //    void compute_score(State& new_state, const State& parent_state,
    //            const Graph::Vertex& merged_vertex) const
    //    {
    //        // we don't use h here

    //    }

    //    void notify_pushed_state(size_t state_index)
    //    {
    //        heap_.push_back(state_index);
    //        std::push_heap(heap_.begin(), heap_.end(), cmp_);
    //    }

    //    size_t next_state_to_expand()
    //    {
    //        std::pop_heap(heap_.begin(), heap_.end(), cmp_);
    //        size_t state_index = heap_.back();
    //        heap_.pop_back();
    //        return state_index;
    //    }
    //};

    bool OutputCmp::operator()(size_t i, size_t j) const
    { return this->operator()(search.states_[i], search.states_[j]); }

    //bool RobustnessCmp::operator()(size_t i, size_t j) const
    //{ return this->operator()(search.states_[i], search.states_[j]); }

} /* namespace veritas */

#endif // VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP
