/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_SEARCH_HPP
#define VERITAS_GRAPH_SEARCH_HPP

#include "domain.hpp"
#include "new_tree.hpp"
#include "new_graph.hpp"
#include <iostream>
#include <chrono>
#include <map>

namespace veritas {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    class GraphSearch;

    struct Snapshot {
        double time = 0.0;
        size_t num_steps = 0;
        size_t num_solutions = 0;
        size_t num_states = 0;
        FloatT eps = 0.0;
        std::tuple<FloatT, FloatT, FloatT> bounds = {-FLOATT_INF, FLOATT_INF, FLOATT_INF}; // lo, up_a, up_ara
    };

    struct State {
        size_t parent; // index into GraphSearch::states_
        FloatT g, h;
        BoxRef box;

        int next_indep_set; // index of the NEXT indep_set to expand into
        bool is_expanded;

        size_t hash;

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
            << "   - hash: " << s.hash << std::endl
            << "}";
    }

    struct StateCmp {
        const GraphSearch& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;

        inline bool operator()(const State& a, const State& b) const
        { return a.fscore(eps) < b.fscore(eps); }
    };

    struct SolutionRef {
        size_t state_index;
        FloatT eps;
        double time;
    };

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
        std::multimap<size_t, size_t> visited_; // hash -> state_index
        StateCmp a_cmp_, ara_cmp_;
        FloatT eps_increment_;
        time_point start_time_;
        std::vector<SolutionRef> solutions_; // indices into states_
        size_t num_steps_;
        double last_eps_increment_, avg_eps_update_time_;
        
    public:
        friend StateCmp;

        std::vector<Snapshot> snapshots;

        GraphSearch(const AddTree& at, int merge = 0)
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
            if (merge > 1)
                g_.merge(merge, 1.000);

            State s = {
                0, // parent
                0.0, // g --> add base score only in solution to avoid negative numbers
                0.0, // h
                BoxRef::null_box(), // box
                0, // next_indep_set
                false, // is_expanded
                0, // hash
            };
            find_indep_sets_not_added(0, workspace_.indep_sets);
            compute_heuristic(s, -1);
            push_state(std::move(s));
            push_snapshot();
        }

        void prune_by_box(const BoxRef& box)
        {
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
            //std::cout << " -> hash history:" << std::endl;
            //size_t si = state_index;
            //while (si != 0)
            //{
            //    std::cout << "    - " << states_[si].hash << std::endl;
            //    si = states_[si].parent;
            //}

            if (states_[state_index].next_indep_set == -1)
            {
                std::cout << "SOLUTION "
                    << " eps=" << state_eps
                    << " output=" << (states_[state_index].g+at_.base_score)
                    << std::endl;
                push_solution(state_index, state_eps);
                update_eps(eps_increment_);
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
                        std::cout << "UPDATE PREVOUS SOLUTION " << s1.fscore(state_eps)
                            << " > " << s0.fscore(state_eps)
                            << " (" << state_eps << " ==? " << ara_cmp_.eps << ")";
                        sol.eps = state_eps;
                        update_eps(eps_increment_);
                        std::cout << " new eps=" << ara_cmp_.eps << std::endl;
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
                if (solutions_.size() > 0 && solutions_[0].eps == 1.0)
                {
                    std::cout << "step_for: stopping early because optimal solution found." << std::endl;
                    break;
                }

                double dur = time_since_start() - start;
                done = steps(num_steps);
                if (dur >= num_seconds)
                    break;
            }

            return done;
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
        void push_state(State&& state)
        {
            size_t state_index = states_.size();
            states_.push_back(std::move(state));

            //std::cout<< "adding " << state_index << " " << states_[state_index] << std::endl;

            push_to_heap(a_heap_, state_index, a_cmp_);

            if (ara_cmp_.eps < 1.0)
                push_to_heap(ara_heap_, state_index, ara_cmp_);

            // push state's hash to visited map
            visited_.insert({states_[state_index].hash, state_index});
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
                else
                {
                    std::cout << "- updating solution eps from " << sol1.eps;
                    sol1.eps = std::max(sol1.eps, eps);
                    std::cout << " to " <<  std::max(solutions_[i-1].eps, eps)
                        << " with s1.g=" << states_[sol1.state_index].g << " >= " << states_[state_index].g << std::endl;
                }
            }
        }

        bool in_visited(const State& state) const
        {
            return false; // does not seem to be necessary, need to proof this

            //size_t state_hash = hash(state); // buffer this
            auto r = visited_.equal_range(state.hash);

            for (auto it = r.first; it != r.second; ++it)
                if (is_same_state(states_[it->second], state))
                    return true;

            return false;
        }

        bool is_same_state(const State& s0, const State& s1) const
        {
            std::cout << " (!) hash collision " << std::endl;
            std::cout << "   - ";
            print_state(std::cout, s0);
            std::cout << std::endl << "   - ";
            print_state(std::cout, s1);
            std::cout << std::endl;

            // compare boxes
            // maybe possible? different paths, but same box -> should be very
            // rare, and they are effectively the same -> previous still is still able to expand
            bool is_same = s0.box == s1.box;

            return is_same;
        }

        void expand(size_t state_index)
        {
            find_indep_sets_not_added(state_index, workspace_.indep_sets);

            states_[state_index].is_expanded = true;
            BoxRef state_box = states_[state_index].box;
            FloatT g = states_[state_index].g;
            size_t prev_hash = states_[state_index].hash;
            int next_indep_set = states_[state_index].next_indep_set;

            // expand by adding vertexes from `next_indep_set`
            Graph::IndepSet set = g_.get_vertices(next_indep_set);
            int num_vertices = static_cast<int>(set.size());
            for (int vertex = 0; vertex < num_vertices; ++vertex) // filled by `find_indep_sets_not_added`
            {
                const Graph::Vertex& v = set[vertex];
                if (v.box.overlaps(state_box))
                {
                    combine_boxes(v.box, state_box, true, workspace_.box);
                    BoxRef box = BoxRef(store_.store(workspace_.box, remaining_mem_capacity()));
                    workspace_.box.clear();
                    State new_state = {
                        state_index, // parent
                        g+v.output,
                        0.0, // heuristic, set later, after `in_visited` check, in `compute_heuristic`
                        box,
                        -1, // indep_sets, set later with h in `compute_heuristic`
                        false, // is_expanded
                        hash(prev_hash, next_indep_set, vertex),
                    };
                    if (!in_visited(new_state))
                    {
                        compute_heuristic(new_state, next_indep_set);
                        if (!std::isinf(new_state.h))
                        {
                            //std::cout << "pushing with hash " << prev_hash << " -> " << new_state.hash
                            //    << " (" << next_indep_set << ", " << vertex << ")"
                            //    << std::endl;
                            push_state(std::move(new_state));
                        }
                        else std::cout << " -> inf h, skipping" << std::endl;
                    }
                    else
                    {
                        std::cout << "ALREADY SEEN ";
                        print_state(std::cout, new_state);
                        std::cout << std::endl;
                    }
                }
            }
        }

        // assume valid state for parent state in `workspace_.indep_sets`
        void compute_heuristic(State& new_state, int skip_indep_set) const
        {
            // set next indep_set to expand into
            // if it stays -1, then this is a solution state (nothing more to expand into)
            new_state.next_indep_set = -1;

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
            buffer.resize(g_.num_independent_sets(), -1);
            if (s.hash == 0) return; // => state_index == 0

            size_t state_index = s.parent;
            while (true)
            {
                const State& s = states_[state_index];
                Graph::IndepSet set = g_.get_vertices(s.next_indep_set);
                int num_vertices = static_cast<int>(set.size());
                for (int vertex = 0 ; vertex < num_vertices; ++vertex)
                    if (set[vertex].box.overlaps(s.box))
                        buffer[s.next_indep_set] = set[vertex].leaf_id;

                if (state_index == 0) break;

                state_index = s.parent;
            }
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
        // generate a hash that is (hopefully) unique for each state
        // state with state_index 0 has hash 0
        // order in which (indep_sets, vertex) pairs are added does not matter
        size_t hash(size_t h, int indep_set, int vertex) const
        {
            size_t a = (67'280'421'310'721 * (std::hash<int>()(indep_set) + 29))
                + (999'999'000'001 * (std::hash<int>()(vertex) + 71));
            h = h ^ a;
            //std::cout << "  hash round: " << indep_set << ", " << vertex << " => " << h << std::endl;
            return h;
        }

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
