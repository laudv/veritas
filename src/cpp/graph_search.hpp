/**
 * \file graph_search.hpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP
#define VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP

#include "domain.hpp"
#include "tree.hpp"
#include "graph.hpp"
//#include "constraints.hpp"
#include <iostream>
#include <chrono>
#include <map>

#include <iomanip>

namespace veritas {

    /** \private */
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    /** \private */
    struct State {
        FloatT g, h, delta;
        BoxRef box;

        int indep_set;
        bool is_expanded;
    };

    static
    std::ostream&
    operator<<(std::ostream& strm, const State& s)
    {
        return strm
            << "State {" << std::endl
            << "   - g, h: " << s.g << ", " << s.h << std::endl
            << "   - delta: " << s.delta << std::endl
            << "   - box: " << s.box << std::endl
            << "   - indep_set: " << s.indep_set << std::endl
            << "   - expanded?: " << s.is_expanded << std::endl
            << "}";
    }

    /** \private */
    struct SolutionRef {
        size_t state_index;
        FloatT eps;
        double time;
    };

    /** A full (sub)optimal solution. */
    struct Solution {
        size_t state_index;
        size_t solution_index;
        FloatT eps;
        FloatT delta;
        FloatT output;
        std::vector<NodeId> nodes; // one leaf node id per tree in addtree
        Box box;
        double time;
    };

    static
    std::ostream&
    operator<<(std::ostream& strm, const Solution& s)
    {
        strm
            << "Solution {" << std::endl
            << "   - state, solution index: " << s.state_index << ", " << s.solution_index << std::endl
            << "   - output: " << s.output << std::endl
            << "   - eps, delta: " << s.eps << ", " << s.delta << std::endl
            << "   - nodes: ";

        if (s.nodes.size() > 0)
        {
            strm << s.nodes.at(0);
            for (auto it = s.nodes.begin()+1; it != s.nodes.end(); ++it)
                strm << ", " << *it;
        }

        return strm
            << std::endl
            << "   - box: " << BoxRef(s.box) << std::endl
            << "   - time: " << s.time << std::endl
            << "}";
    }

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
        FloatT eps;

        bool operator()(size_t i, size_t j) const;
        inline bool operator()(const State& a, const State& b) const;
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
     *  
     *  - Derived::stop_conditions_met
     *      * call GraphSearch::stop_conditions_met_
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
        size_t mem_capacity_;
        time_point start_time_;
        size_t num_steps_;
        const CmpT& solution_cmp_;

        friend OutputCmp;
        friend RobustnessCmp;

        GraphSearch(const AddTree& at, const CmpT& solution_cmp)
            : at_(at.neutralize_negative_leaf_values())
            , g_(at_)
            , mem_capacity_(size_t(1024)*1024*1024)
            , start_time_{std::chrono::system_clock::now()}
            , num_steps_{0}
            , solution_cmp_(solution_cmp)
        {}

        /** call in Derived constructor */
        void init()
        {
            State state = {
                0.0, 0.0, // g, h
                0.0, // delta
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
        size_t stop_when_num_solutions_equals = 999'999'999; /**< disabled by default */
        int break_steps_when_n_new_solutions = 1;

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
            for (size_t i = 0; i < num_steps
                    && num_sol + break_steps_when_n_new_solutions > num_solutions(); ++i)
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

            int next_indep_set = states_[state_index].indep_set + 1;
            Graph::IndepSet set = g_.get_vertices(next_indep_set);
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

        bool stop_conditions_met_() const
        {
            if (solutions_.size() >= stop_when_num_solutions_equals)
            {
                std::cout << "stop_conditions_met: stopping early: "
                    << num_solutions() << " >= "
                    << stop_when_num_solutions_equals
                    << " solutions found"
                    << std::endl;
                return true;
            }

            return false;
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
        size_t num_steps() const { return num_steps_; }
        size_t num_solutions() const { return solutions_.size(); }
        void set_mem_capacity(size_t bytes) { mem_capacity_ = bytes; }
        size_t remaining_mem_capacity() const
        { return mem_capacity_ - store_.get_mem_size(); }

        /** Seconds since the construction of the search */
        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time_).count() * 1e-6;
        }

        Solution get_solution(size_t solution_index) const
        {
            auto&& [state_index, eps, time] = solutions_.at(solution_index);
            const State& state = states_[state_index];

            std::vector<NodeId> node_ids;
            find_node_ids(state, node_ids);

            return {
                state_index,
                solution_index,
                eps,
                state.delta,
                at_.base_score + state.g,
                node_ids, // copy
                {state.box.begin(), state.box.end()},
                time,
            };
        }


    private:

        void construct_and_push_states(size_t parent_state_index, const Graph::Vertex& v)
        {
            auto push_workspace_box_fun = [this, parent_state_index, &v](Box& b){
                int indep_set = states_[parent_state_index].indep_set + 1;
                BoxRef box = BoxRef(store_.store(b, remaining_mem_capacity()));
                State new_state = {
                    0.0, // g
                    0.0, // h heuristic, set later
                    0.0, // delta
                    box,
                    indep_set,
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

        void find_node_ids(const State& s, std::vector<NodeId>& buffer) const
        {
            buffer.clear();
            for (const Graph::IndepSet& set : g_)
                for (const Graph::Vertex& v : set)
                    if (v.box.overlaps(s.box))
                    { buffer.push_back(v.leaf_id); break; }
        }

    protected:
        FloatT compute_output_heuristic(const State& state) const
        {
            // compute heuristic
            FloatT h = 0.0;
            for (size_t indep_set = state.indep_set+1; indep_set < g_.num_independent_sets(); ++indep_set)
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
            std::vector<FloatT> d0, d1;
            int prev_indep_set = -1;
            for (size_t indep_set = state.indep_set+1; indep_set < g_.num_independent_sets(); ++indep_set)
            {
                // fill d0 with the first indep_set
                if (d0.empty())
                {
                    for (const auto& v : g_.get_vertices(indep_set))
                    {
                        if (v.box.overlaps(state.box))
                            d0.push_back(v.output);
                        else
                            d0.push_back(-FLOATT_INF);
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
                            for (size_t j = 0; j < d0.size(); ++j)
                            {
                                const auto& v0 = set0[j];
                                if (!std::isinf(d0[j]) && v0.box.overlaps(v1.box))
                                    max = std::max(max, d0[j]);
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

            // find max of d0 --> thats the heuristic value
            if (state.indep_set+1 < static_cast<int>(g_.num_independent_sets()))
            {
                FloatT max = -FLOATT_INF;
                for (auto v : d0)
                    max = std::max(max, v);
                return max;
            }
            return 0.0; // final state has no heuristic
        }

        FloatT compute_delta(const State& state, const std::vector<FloatT>& example) const
        {
            FloatT delta = state.delta;
            for (auto &&[feat_id, dom] : state.box)
            {
                FloatT x = example[feat_id];
                if (!dom.contains(x))
                {
                    FloatT d = std::min(std::abs(dom.lo - x), std::abs(dom.hi - x));
                    //std::cout << "feature " << feat_id << ": " << x << ", " << dom << " -> " << d << std::endl;
                    delta = std::max(delta, d);
                }
            }
            return delta;
        }
    }; // class GraphSearch

    class GraphOutputSearch : public GraphSearch<GraphOutputSearch, OutputCmp> {
        std::vector<size_t> a_heap_, ara_heap_; // indexes into states_
        OutputCmp a_cmp_, ara_cmp_;
        FloatT eps_increment_;
        double last_eps_increment_, avg_eps_update_time_;

        using CmpT = OutputCmp;

    public:
        friend GraphSearch<GraphOutputSearch, OutputCmp>;

        // SETTINGS
        bool use_dynprog_heuristic = false;

        FloatT stop_when_solution_eps_equals = 1.0; // default: when optimal
        FloatT stop_when_up_bound_less_than = -FLOATT_INF; // default: disabled
        FloatT stop_when_solution_output_greater_than = FLOATT_INF;

        std::vector<Snapshot> snapshots;

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
            if (use_dynprog_heuristic)
                new_state.h = compute_output_heuristic_dynprog(new_state);
            else
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

            //std::cout << "solution fscore=" << (g_.base_score+a_cmp_.fscore(states_[state_index]))
            //    << ", eps=" << sol.eps << ", nsteps=" << num_steps_ << std::endl;

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
            bool sol_updated = false;
            if (num_solutions() > 0 && ara_heap_.size() > 0)
            {
                const State& s0 = states_[state_index];
                for (SolutionRef& sol : solutions_)
                {
                    const State& s1 = states_[sol.state_index];
                    std::cout << "UPDATE SOL? "
                        << (ara_cmp_.fscore(s1)+g_.base_score)
                        << " > " << (ara_cmp_.fscore(s0)+g_.base_score)
                        << " (" << sol.eps << " ==? " << ara_cmp_.eps << ") " << sol.state_index
                        << ", " << state_index
                        << std::endl;
                    if (sol.eps != 1.0 && ara_cmp_(s0, s1))
                    {
                        //std::cout << "UPDATE PREVIOUS SOLUTION "
                        //    << (ara_cmp_.fscore(s1)+g_.base_score)
                        //    << " > " << (ara_cmp_.fscore(s0)+g_.base_score)
                        //    << " (" << sol.eps << " ==? " << ara_cmp_.eps << ")"
                        //    << std::endl;
                        sol.eps = ara_cmp_.eps;
                        sol_updated = true;
                        //std::cout << " new eps=" << ara_cmp_.eps << std::endl;
                    }
                    else break;
                }
            }
            if (sol_updated)
                update_eps(eps_increment_);

            // if finding another suboptimal solution takes too long,
            // decrease eps, halve eps_increment_
            double t = time_since_start();
            if (last_eps_increment_ > 0.0 && (t-last_eps_increment_) > 20*avg_eps_update_time_)
            {
                //std::cout << "HALVING eps_increment_ to " << eps_increment_/2.0
                //    << " avg t: " << avg_eps_update_time_
                //    << " t: " << (t-last_eps_increment_);
                eps_increment_ = std::max(0.001, eps_increment_/2.0);
                update_eps(-eps_increment_);
                //std::cout
                //    << " (eps=" << ara_cmp_.eps << ")"
                //    << std::endl;
            }
        }
        
    public:
        bool step()
        {
            size_t state_index;
            while (true)
            {
                if (a_heap_.empty() && ara_heap_.empty())
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

        bool stop_conditions_met() const
        {
            if (stop_conditions_met_())
                return true;

            auto [lo, up_a, up_ara] = current_bounds_with_base_score();
            if (std::min(up_a, up_ara) < stop_when_up_bound_less_than)
            {
                std::cout << "stop_conditions_met: stopping early: "
                    << "upper bound " << stop_when_up_bound_less_than
                    << " reached (" << up_a << ", " << up_ara << ")"
                    << std::endl;
                return true;
            }

            if (solutions_.size() > 0 && solutions_[0].eps == stop_when_solution_eps_equals)
            {
                std::cout << "stop_conditions_met: stopping early: "
                    << "solution with eps " << stop_when_solution_eps_equals
                    << " found." << std::endl;
                return true;
            }

            if (solutions_.size() > 0 && a_cmp_.fscore(states_[solutions_.front().state_index])
                    > stop_when_solution_output_greater_than)
            {
                std::cout << "stop_conditions_met: stopping early: "
                    << "solution found with output greater than "
                    << stop_when_solution_output_greater_than
                    << std::endl;
                return true;
            }

            return false;
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
            snapshots.push_back({
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

    public:

        /** ARA* lower, A* upper, ARA* upper */
        std::tuple<FloatT, FloatT, FloatT> current_bounds_with_base_score() const
        {
            auto &&[lo, up_a, up_ara] = current_bounds();
            return {lo+at_.base_score, up_a+at_.base_score, up_ara+at_.base_score};
        }

        FloatT get_eps() const { return ara_cmp_.eps; }

        void set_eps(FloatT eps)
        {
            std::cout << "UPDATING EPS " << ara_cmp_.eps << " -> ";
            ara_cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps));
            std::cout << ara_cmp_.eps << std::endl;
            if (ara_cmp_.eps < 1.0)
                std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            else
                ara_heap_.clear();
        }

        void set_eps_increment(FloatT incr) { eps_increment_ = incr; }

        void update_eps(FloatT added_value) // this updates time, set_eps does not
        {
            set_eps(ara_cmp_.eps + added_value);

            double t = time_since_start();
            double time_since_previous_incr = t - last_eps_increment_;
            if (time_since_previous_incr*10 < avg_eps_update_time_)
            {
                eps_increment_ *= 2;
                //std::cout << "DOUBLING eps_increment_ to " << eps_increment_
                //    << " avg t: " << avg_eps_update_time_
                //    << " t: " << (t - last_eps_increment_)
                //    << std::endl;
            }
            last_eps_increment_ = t;
            avg_eps_update_time_ = 0.2*avg_eps_update_time_ + 0.8*time_since_previous_incr;
        }



    };

    /** Minimize the delta between the given example and the generated solution. */
    class GraphRobustnessSearch : public GraphSearch<GraphRobustnessSearch, RobustnessCmp> {

        std::vector<size_t> heap_; // indexes into states_
        RobustnessCmp cmp_;

        std::vector<FloatT> example_;
        FloatT max_delta_;

        using CmpT = RobustnessCmp;

    public:
        friend GraphSearch<GraphRobustnessSearch, RobustnessCmp>;

        // SETTINGS
        FloatT output_threshold = 0.0; /**< Reject states that cannot exceed this output */

        GraphRobustnessSearch(const AddTree& at, const std::vector<FloatT>& example, FloatT max_delta)
            : GraphSearch(at, cmp_)
            , cmp_{*this, 1.0}
            , example_(example)
            , max_delta_(max_delta)
        {
            Box box;
            for (FeatId feat_id = 0; feat_id < static_cast<int>(example.size()); ++feat_id)
            {
                FloatT x = example_[feat_id];
                box.push_back({feat_id, {x - max_delta_, x + max_delta_}});
            }
            prune_by_box(box);
            init();
        }

    private:
        void compute_score(State& new_state, const State& parent_state,
                const Graph::Vertex& merged_vertex) const
        {
            // state.g contains output so far
            // state.h contains delta estimate
            new_state.g = parent_state.g + merged_vertex.output;
            //if (use_dynprog_heuristic)
            //    new_state.h = compute_output_heuristic_dynprog(new_state);
            //else
                new_state.h = compute_output_heuristic(new_state);

            // compute L0 norm given the new vertex
            new_state.delta = compute_delta(new_state, example_);
        }

        void push_state(State&& state)
        {
            // reject states with output certainly < 0 (label not flipped)
            if (g_.base_score + state.g + state.h <= output_threshold)
            { 
                //std::cout << "rejecting state g+h=" << g_.base_score+state.g+state.h
                //    << ", g=" << state.g
                //    << ", h=" << state.h
                //    << " at depth " << state.indep_set << std::endl;
            }
            else
            {
                size_t state_index = push_state_(std::move(state));
                heap_.push_back(state_index);
                std::push_heap(heap_.begin(), heap_.end(), cmp_);
            }
        }

        void push_solution(size_t state_index)
        {
            if (states_[state_index].delta > 0.0)
            {
                size_t solution_index = push_solution_(state_index);
                solutions_[solution_index].eps = cmp_.eps;
                //std::cout << "accepted solution " << states_[state_index].box
                //    << ", delta " << states_[state_index].h << std::endl;
            }
            else
            {
                //std::cout << "rejected solution " << states_[state_index].box
                //    << ", delta " << states_[state_index].h << std::endl;
            }

        }

        void expand(size_t state_index)
        {
            expand_(state_index);
        }

    public:
        bool step()
        {
            if (heap_.empty())
                return true;
            std::pop_heap(heap_.begin(), heap_.end(), cmp_);
            size_t state_index = heap_.back();
            heap_.pop_back();

            //const State& s = states_[state_index];
            //std::cout << "step " << s.box << " g=" << s.g << ", delta=" << s.h
            //      << " indep_set=" << s.indep_set << std::endl;

            step_(state_index);
            return false; // not done
        }

        bool steps(size_t num_steps)
        {
            bool done = steps_(num_steps);
            //push_snapshot();
            return done;
        }

        bool stop_conditions_met() const
        {
            if (stop_conditions_met_())
                return true;
            return false;
        }

        void set_eps(FloatT eps) 
        {
            cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps));
            std::make_heap(heap_.begin(), heap_.end(), cmp_);
        }
        FloatT get_eps() const { return cmp_.eps; }
    };

    inline bool OutputCmp::operator()(size_t i, size_t j) const
    { return this->operator()(search.states_[i], search.states_[j]); }

    inline bool RobustnessCmp::operator()(size_t i, size_t j) const
    { return this->operator()(search.states_[i], search.states_[j]); }

    // return TRUE if first is _less_ than second (ie. you want second higher
    // up in the max heap)
    bool RobustnessCmp::operator()(const State& a, const State& b) const
    {
        // smaller delta (in h) is better, promote deeper by discounting h
        FloatT num_indep_sets = search.g_.num_independent_sets();
        FloatT ad = (eps + (1.0-eps)*(1.0 - a.indep_set/num_indep_sets)) * a.delta;
        FloatT bd = (eps + (1.0-eps)*(1.0 - b.indep_set/num_indep_sets)) * b.delta;

        //bool x = (ad == bd) ? (a.g+a.h) < (b.g+b.h) : ad > bd;
        //std::cout << std::setprecision(4)
        //    << "discount " << (eps + (1.0-eps)*(1.0 - a.indep_set/num_indep_sets))
        //    << " at " << a.indep_set
        //    << " delta=" << a.delta
        //    << " g+h=" << (a.g+a.h)
        //    << " -> " << ad
        //    << std::endl;
        //std::cout << std::setprecision(4)
        //    << " ------- " << (eps + (1.0-eps)*(1.0 - b.indep_set/num_indep_sets))
        //    << " at " << b.indep_set
        //    << " delta=" << b.delta
        //    << " g+h=" << (b.g+b.h)
        //    << " -> " << bd
        //    << "   ==> " << (x ? "b wins" : "a wins")
        //    << std::endl;

        return (ad == bd) ? (a.g+a.h) < (b.g+b.h) : ad > bd;
        //return a.delta > b.delta;
    }

} /* namespace veritas */

#endif // VERITAS_GRAPH_ROBUSTNESS_SEARCH_HPP
