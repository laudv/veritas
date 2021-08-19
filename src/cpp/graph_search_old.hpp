/**
 * \file graph_search.hpp
 *
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_SEARCH_HPP
#define VERITAS_GRAPH_SEARCH_HPP

#include "domain.hpp"
#include "tree.hpp"
#include "graph.hpp"
#include "constraints.hpp"
#include <iostream>
#include <chrono>
#include <map>

namespace veritas {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    class GraphSearch;

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

    /** \private */
    struct State {
        size_t parent; // index into GraphSearch::states_
        FloatT g, h;
        BoxRef box;

        int next_indep_set; // index of the NEXT indep_set to expand into
        bool is_expanded;

        inline FloatT fscore(FloatT eps=1.0) const { return g + eps*h; }
    };

    std::ostream&
    operator<<(std::ostream& strm, const State& s)
    {
        return strm
            << "State {" << std::endl
            << "   - parent: " << s.parent << std::endl
            << "   - g, h: " << s.g << ", " << s.h << std::endl
            << "   - box: " << s.box << std::endl
            << "   - next_indep_set: " << s.next_indep_set << std::endl
            << "   - expanded?: " << s.is_expanded << std::endl
            << "}";
    }

    /** \private */
    struct StateCmp {
        const GraphSearch& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;

        inline bool operator()(const State& a, const State& b) const
        { return a.fscore(eps) < b.fscore(eps); }
    };

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
        FloatT output;
        std::vector<NodeId> nodes; // one leaf node id per tree in addtree
        Box box;
        double time;
    };

    std::ostream&
    operator<<(std::ostream& strm, const Solution& s)
    {
        strm
            << "Solution {" << std::endl
            << "   - state, solution index: " << s.state_index << ", " << s.solution_index << std::endl
            << "   - output, eps: " << s.output << ", " << s.eps << std::endl
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

    class GraphSearch {
        AddTree at_;
        Graph g_;

        BlockStore<DomainPair> store_;
        mutable struct {
            Box box;
            std::vector<int> indep_sets;
            std::vector<int> vertices;
        } workspace_;

        std::vector<State> states_;
        std::vector<size_t> a_heap_, ara_heap_; // indexes into states_
        StateCmp a_cmp_, ara_cmp_;
        FloatT eps_increment_;
        time_point start_time_;
        std::vector<SolutionRef> solutions_; // indices into states_
        size_t num_steps_;
        double last_eps_increment_, avg_eps_update_time_;
        
    public:
        friend StateCmp;

        std::vector<Snapshot> snapshots;
        std::unique_ptr<ConstraintPropagator> constr_prop;

        // SETTINGS
        bool use_dynprog_heuristic = false;

        FloatT stop_when_solution_eps_equals = 1.0; // default: when optimal
        size_t stop_when_num_solutions_equals = 999'999'999; // default: de facto disabled
        FloatT stop_when_up_bound_less_than = -FLOATT_INF; // default: disabled
        FloatT stop_when_solution_output_greater_than = FLOATT_INF;


        GraphSearch(const AddTree& at)
            : at_(at.neutralize_negative_leaf_values())
            , g_(at_)
            , a_cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.01}
            , eps_increment_{0.01}
            , start_time_{std::chrono::system_clock::now()}
            , num_steps_{0}
            , last_eps_increment_{0.0}
            , avg_eps_update_time_{0.0}
        {
            State s = {
                0, // parent
                0.0, // g --> add base score only in solution to avoid negative numbers
                0.0, // h
                BoxRef::null_box(), // box
                0, // next_indep_set
                false, // is_expanded
            };
            find_indep_sets_not_added(0, workspace_.indep_sets);
            compute_heuristic(s, -1);
            push_state(std::move(s));
            push_snapshot();
        }

        void prune_by_box(const BoxRef& box)
        {
            if (states_.size() > 1)
                throw std::runtime_error("invalid state: pruning with more than 1 state");
            g_.prune_by_box(box, false);
        }

        bool step()
        {
            size_t state_index;
            FloatT state_eps;
            while (true)
            {
                if (a_heap_.empty())
                    return true;

                std::tie(state_index, state_eps) = pop_state();
                if (!states_[state_index].is_expanded)
                    break;
            }
            
            //std::cout << "expanding " << state_index << " ";
            //print_state(std::cout, states_[state_index]);
            //std::cout << " eps=" << state_eps;
            //std::cout << " fscore=" << states_[state_index].fscore(state_eps) << std::endl;

            if (states_[state_index].next_indep_set == -1)
            {
                //std::cout << "SOLUTION "
                //    << " eps=" << state_eps
                //    << " output=" << (states_[state_index].g+at_.base_score)
                //    << std::endl;
                push_solution(state_index, state_eps);
                if (state_eps != 1.0)
                    update_eps(eps_increment_);
                else
                    set_eps(1.0);
            }
            else
            {
                expand(state_index);

                // if the previous best suboptimal solution was still in the ara
                // stack, would it be at the top? if so, increase eps
                if (num_solutions() > 0 && state_eps != 1.0 && ara_heap_.size() > 0)
                {
                    SolutionRef& sol = solutions_[0];
                    const State& s1 = states_[sol.state_index];
                    const State& s0 = states_[state_index];
                    if (sol.eps != 1.0 && s1.fscore(state_eps) > s0.fscore(state_eps))
                    {
                        //std::cout << "UPDATE PREVIOUS SOLUTION " << s1.fscore(state_eps)
                        //    << " > " << s0.fscore(state_eps)
                        //    << " (" << state_eps << " ==? " << ara_cmp_.eps << ")";
                        sol.eps = state_eps;
                        update_eps(eps_increment_);
                        //std::cout << " new eps=" << ara_cmp_.eps << std::endl;
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

            ++num_steps_;
            return false;
        }

        bool steps(size_t num_steps)
        {
            bool done = false;
            size_t num_sol = num_solutions();
            for (size_t i = 0; i < num_steps && num_sol == num_solutions(); ++i)
            {
                if (step())
                {
                    done = true;
                    break;
                }
            }
            push_snapshot();

            // TRYING OUT: cap heuristic based on ARA* upper bound
            //auto&& [lo, up_a, up_ara] = current_bounds();
            //if ((up_a - up_ara) / up_ara > 1e-5)
            //{
            //    double start = time_since_start();
            //    size_t count = 0;
            //    for (size_t state_index : a_heap_)
            //    {
            //        State& s = states_[state_index];
            //        if (s.g + s.h > up_ara)
            //        {
            //            s.h = up_ara - s.g;
            //            ++count;
            //        }
            //    }
            //    for (size_t state_index : ara_heap_)
            //    {
            //        State& s = states_[state_index];
            //        if (s.g + s.h > up_ara)
            //        {
            //            s.h = up_ara - s.g;
            //            ++count;
            //        }
            //    }
            //    std::make_heap(a_heap_.begin(), a_heap_.end(), a_cmp_);
            //    std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            //    auto&& [lo, new_up_a, up_ara] = current_bounds();
            //    std::cout << "UPDATED! " << count << ": " << up_a << " -> " << new_up_a << " == " << up_ara
            //        << " (size of heap " << a_heap_.size() << ")"
            //        << " in " << (time_since_start() - start) << " seconds\n";
            //}

            return done;
        }

        bool step_for(double num_seconds, size_t num_steps)
        {
            double start = time_since_start();
            bool done = false;

            while (!done)
            {
                if (stop_conditions_met())
                    break;

                double dur = time_since_start() - start;
                done = steps(num_steps);
                if (dur >= num_seconds)
                    break;
            }

            return done;
        }

        bool stop_conditions_met() const
        {
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

            if (solutions_.size() >= stop_when_num_solutions_equals)
            {
                std::cout << "stop_conditions_met: stopping early: "
                    << num_solutions() << " >= "
                    << stop_when_num_solutions_equals
                    << " solutions found"
                    << std::endl;
                return true;
            }

            if (solutions_.size() > 0 && states_[solutions_.front().state_index].fscore()
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

        //void solve_subset(size_t num_states, size_t num_steps)
        //{
        //    std::vector<size_t> ara_heap, a_heap, tmp;
        //    for (size_t i = 0; i < num_states; ++i)
        //        tmp.push_back(std::get<0>(pop_state()));

        //    // swap ARA heaps
        //    std::swap(a_heap_, a_heap);
        //    std::swap(ara_heap_, ara_heap);

        //    // push states we will focus on
        //    for (size_t state_index : tmp)
        //    {
        //        push_to_heap(a_heap_, state_index, a_cmp_);
        //        push_to_heap(ara_heap_, state_index, ara_cmp_);
        //    }

        //    steps(num_steps);

        //    while (

        //    // swap back
        //    std::swap(a_heap_, a_heap);
        //    std::swap(ara_heap_, ara_heap);
        //}

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

        size_t num_solutions() const { return solutions_.size(); }

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
                at_.base_score + state.g,
                node_ids, // copy
                {state.box.begin(), state.box.end()},
                time,
            };
        }

        size_t num_states() const
        {
            return states_.size();
        }

        std::tuple<size_t, size_t> heap_size() const
        {
            return {a_heap_.size(), ara_heap_.size()};
        }

        /** seconds since the construction of the search */
        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time_).count() * 1e-6;
        }

        FloatT get_eps() const { return ara_cmp_.eps; }

        void set_eps(FloatT eps)
        {
            ara_cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps));
            if (ara_cmp_.eps < 1.0)
                std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            else
                ara_heap_.clear();
        }

        void set_eps_increment(FloatT incr)
        {
            eps_increment_ = incr;
        }

        /** ARA* lower, A* upper, ARA* upper */
        std::tuple<FloatT, FloatT, FloatT> current_bounds_with_base_score() const
        {
            auto &&[lo, up_a, up_ara] = current_bounds();
            return {lo+at_.base_score, up_a+at_.base_score, up_ara+at_.base_score};
        }



    private:
        //void push_state(State&& state)
        //{
        //    if (constr_prop)
        //    {
        //        constr_prop->check(workspace_.box, [this, &state](const Box& box) {
        //            State state1 = state;
        //            state1.box = BoxRef(store_.store(box, remaining_mem_capacity()));
        //            std::cout << "push_box_fun: pushing state " << state1 << std::endl;
        //            push_state_aux(std::move(state1));
        //        });
        //    }
        //    else push_state_aux(std::move(state));
        //}

        void push_state(State&& state)
        {
            size_t state_index = states_.size();
            states_.push_back(std::move(state));

            //std::cout<< "adding " << state_index << " " << states_[state_index] << std::endl;

            push_to_heap(a_heap_, state_index, a_cmp_);

            if (ara_cmp_.eps < 1.0)
                push_to_heap(ara_heap_, state_index, ara_cmp_);
        }

        void push_to_heap(std::vector<size_t>& heap, size_t state_index, const StateCmp& cmp)
        {
            heap.push_back(state_index);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }

        /** returns {state_index, eps with which state was selected} */
        std::tuple<size_t, FloatT> pop_state()
        {
            size_t state_index;
            FloatT eps;
            if (!ara_heap_.empty() && num_steps_%2 == 1)
            {
                state_index = pop_from_heap(ara_heap_, ara_cmp_);
                eps = ara_cmp_.eps;
            }
            else
            {
                state_index = pop_from_heap(a_heap_, a_cmp_);
                eps = a_cmp_.eps;
            }
            return {state_index, eps};
        }

        size_t pop_from_heap(std::vector<size_t>& heap, const StateCmp& cmp)
        {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            size_t state_index = heap.back();
            heap.pop_back();
            return state_index;
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
            FloatT up_a = states_[a_heap_.front()].fscore();
            return {lo, up_a, up_ara};
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

        void push_solution(size_t state_index, FloatT eps)
        {
            states_[state_index].is_expanded = true;
            solutions_.push_back({ state_index, eps, time_since_start() });

            // sort solutions
            size_t i = solutions_.size()-1;
            for (; i > 0; --i)
            {
                SolutionRef& sol1 = solutions_[i-1];
                SolutionRef& sol2 = solutions_[i];
                if (states_[sol1.state_index].g < states_[sol2.state_index].g)
                {
                    std::swap(sol1, sol2);
                }
                else if (sol1.eps < eps)
                {
                    //std::cout << "- updating solution eps from " << sol1.eps;
                    sol1.eps = std::max(sol1.eps, eps);
                    //std::cout << " to " <<  std::max(solutions_[i-1].eps, eps)
                    //    << " with s1.g=" << states_[sol1.state_index].g << " >= "
                    //    << states_[state_index].g << std::endl;
                }
            }
        }

        void expand(size_t state_index)
        {
            find_indep_sets_not_added(state_index, workspace_.indep_sets);

            states_[state_index].is_expanded = true;
            BoxRef parent_box = states_[state_index].box;
            int next_indep_set = states_[state_index].next_indep_set;

            // expand by adding vertexes from `next_indep_set`
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

        void construct_and_push_states(size_t parent_state_index, const Graph::Vertex& v)
        {
            auto push_workspace_box_fun = [this, parent_state_index, &v](Box& b){
                FloatT g = states_[parent_state_index].g;
                int next_indep_set = states_[parent_state_index].next_indep_set;

                BoxRef box = BoxRef(store_.store(b, remaining_mem_capacity()));
                State new_state = {
                    parent_state_index,
                    g+v.output,
                    0.0, // heuristic, set later, after `in_visited` check, in `compute_heuristic`
                    box,
                    -1, // indep_sets, set later with h in `compute_heuristic`
                    false, // is_expanded
                };

                compute_heuristic(new_state, next_indep_set);

                if (!std::isinf(new_state.h))
                {
                    push_state(std::move(new_state));
                }
                else std::cout << " -> inf h, skipping" << std::endl;

            };

            if (constr_prop)
            {
                constr_prop->check(workspace_.box, push_workspace_box_fun);
                //constr_prop->print();
            }
            else
            {
                push_workspace_box_fun(workspace_.box);
            }

            workspace_.box.clear();
        }

        void compute_heuristic(State& new_state, int skip_indep_set) const
        {
            // set next indep_set to expand into
            // if it stays -1, then this is a solution state (nothing more to expand into)
            new_state.next_indep_set = -1;

            if (use_dynprog_heuristic)
            {
                compute_heuristic_dynprog(new_state, skip_indep_set);
            }
            else
            {
                compute_heuristic_simple(new_state, skip_indep_set);
            }
        }

        // assume valid state for parent state in `workspace_.indep_sets`
        void compute_heuristic_simple(State& new_state, int skip_indep_set) const
        {
            FloatT maxmax = -FLOATT_INF;
            for (int i : workspace_.indep_sets) // filled by `find_indep_sets_not_added`
            {
                if (i == skip_indep_set) // we just added this one
                    continue;

                FloatT max = -FLOATT_INF;
                Graph::IndepSet set = g_.get_vertices(i);
                for (const auto& v : g_.get_vertices(i))
                    if (v.box.overlaps(new_state.box))
                        max = std::max(max, v.output);

                new_state.h += max;
                if (max > maxmax)
                {
                    new_state.next_indep_set = i; // set next indep_set to expand into
                    maxmax = max;
                }
            }

            // test
            //if (t && new_state.parent != 0)
            //{
            //    GraphSearch s(at_);
            //    s.t = false;
            //    s.set_eps(1.0);
            //    s.prune_by_box(new_state.box);
            //    s.steps(50);
            //    num_steps_ += 50;
            //    auto [lo, up_a, up_ara] = s.current_bounds();
            //    std::cout << "heuristic: " << new_state.fscore() << " <-> " <<
            //        up_a << ", " << (new_state.fscore() < up_a) << std::endl;
            //    new_state.h = up_a-new_state.g;
            //}

            // cap heuristic to ARA* upper bound
            //if (num_solutions() > 0)
            //{
            //    FloatT up_ara = std::get<2>(current_bounds());
            //    if (new_state.fscore() > up_ara)
            //    {
            //        //std::cout << "capping H from " << new_state.h << " to ";
            //        new_state.h = up_ara - new_state.g;
            //        //std::cout << new_state.h << std::endl;
            //    }
            //}
        }

        // assume valid state for parent state in `workspace_.indep_sets`
        void compute_heuristic_dynprog(State& new_state, int skip_indep_set) const
        {
            std::vector<FloatT> d0, d1;
            FloatT maxmax = -FLOATT_INF;

            // fill d0 with first indep_set's output values
            int prev_indep_set = -1;

            // compute heuristic using dynamic programming
            for (size_t i = 0; i < workspace_.indep_sets.size(); ++i)
            {
                FloatT tree_max = -FLOATT_INF;

                int indep_set = workspace_.indep_sets[i];
                if (indep_set == skip_indep_set)
                    continue;

                // fill d0 with the first indep_set
                if (d0.empty())
                {
                    for (const auto& v : g_.get_vertices(indep_set))
                    {
                        if (v.box.overlaps(new_state.box))
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

                        if (v1.box.overlaps(new_state.box))
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

                if (maxmax < tree_max)
                {
                    maxmax = tree_max;
                    new_state.next_indep_set = indep_set;
                }

                prev_indep_set = indep_set;
            }

            // find max of d0 --> thats the heuristic value
            if (new_state.next_indep_set != -1)
            {
                FloatT max = -FLOATT_INF;
                for (auto v : d0)
                    max = (v > max) ? v : max;
                new_state.h = max;
            }
            else // this is a final state with no heuristic estimate
            {
                new_state.h = 0.0;
            }
        }

        //bool t = true;

        void find_indep_sets_not_added(size_t state_index, std::vector<int>& buffer) const
        {
            buffer.clear();
            for (size_t i=0; i < g_.num_independent_sets(); ++i)
                buffer.push_back(i); // push [0, 1, 2, 3, ..., num_independent_sets)

            while (state_index != 0)
            {
                state_index = states_[state_index].parent;
                const State& s = states_[state_index];
                buffer[s.next_indep_set] = -1; // set to 0 to indicate already in state
            }

            buffer.erase(
                std::remove_if(
                    buffer.begin(),
                    buffer.end(),
                    [](int x) { return x == -1; }),
                buffer.end());
        }

        void find_node_ids(const State& s, std::vector<NodeId>& buffer) const
        {
            buffer.clear();

            for (const Graph::IndepSet& set : g_)
                for (const Graph::Vertex& v : set)
                    if (v.box.overlaps(s.box))
                    { buffer.push_back(v.leaf_id); continue; }
        }

        void print_state(std::ostream& o, const State& s) const
        {
            o
                << "State { "
                << "g=" << s.g << ", "
                << "h=" << s.h << ", "
                //<< "box=" << s.box << ", "
                << "leafs=";

            std::vector<NodeId> buf;
            find_node_ids(s, buf);
            for (size_t i = 0; i < g_.num_independent_sets(); ++i)
                if (buf[i] == -1) o << "? ";
                else o << buf[i] << ' ';
                
            //o << "hash=" << s.hash << " }";
        }



    public:

        size_t remaining_mem_capacity() const
        {
            return (size_t(1024)*1024*1024) - store_.get_mem_size();
        }
    }; // class GraphSearch

    bool
    StateCmp::operator()(size_t i, size_t j) const
    { return (*this)(search.states_[i], search.states_[j]); }

} // namespace veritas

#endif // VERITAS_GRAPH_SEARCH_HPP
