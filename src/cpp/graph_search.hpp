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
        std::tuple<FloatT, FloatT> bounds = {-FLOATT_INF, FLOATT_INF};
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
        time_point start_time_;
        std::vector<SolutionRef> solutions_; // indices into states_
        size_t num_steps_;
        
    public:
        friend StateCmp;

        std::vector<Snapshot> snapshots;

        GraphSearch(const AddTree& at)
            : g_(at)
            , a_cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.1}
            , start_time_{std::chrono::system_clock::now()}
            , num_steps_{0}
        {
            State s = {
                0, // parent
                g_.base_score, // g
                0.0, // h
                BoxRef::null_box(), // box
                0, // next_indep_set
                false, // is_expanded
                0, // hash
            };
            find_indep_sets_not_added(0, workspace_.indep_sets);
            compute_heuristic(s, -1);
            push_state(std::move(s));
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
                std::cout << "SOLUTION"
                    << " eps=" << state_eps
                    << " fscore=" << states_[state_index].fscore(state_eps)
                    << " box=" << states_[state_index].box
                    << std::endl;
                push_solution(state_index, state_eps);
                ara_cmp_.eps = std::min(1.0, ara_cmp_.eps + 0.1);
                if (ara_cmp_.eps < 1.0)
                    std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
                else
                    ara_heap_.clear();
            }
            else
            {
                expand(state_index);
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

            snapshots.push_back({
                time_since_start(),
                num_steps_,
                num_solutions(),
                num_states(),
                current_bounds()});

            return done;
        }

        bool step_for(double num_seconds)
        {
            double start = time_since_start();
            bool done = false;

            while (!done)
            {
                double dur = time_since_start() - start;
                done = steps(100);
                if (dur >= num_seconds)
                    break;
            }

            return done;
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
                state.g,
                node_ids, // copy
                {state.box.begin(), state.box.end()},
                time,
            };
        }

        size_t num_states() const
        {
            return states_.size();
        }

        /** seconds since the construction of the search */
        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time_).count() * 1e-6;
        }

        std::tuple<FloatT, FloatT> current_bounds() const
        {
            FloatT lo = FLOATT_INF;
            if (num_solutions() > 0)
                lo = states_[solutions_[0].state_index].g; // best solution so far
            FloatT up = states_[a_heap_.front()].fscore();
            return {lo, up};
        }

        FloatT get_eps() const { return ara_cmp_.eps; }

        void set_eps(FloatT eps)
        { ara_cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps)); }


    private:
        void push_state(State&& state)
        {
            size_t state_index = states_.size();
            states_.push_back(std::move(state));

            //std::cout<< "adding " << state_index << " " << states_[state_index] << std::endl;

            a_heap_.push_back(state_index);
            std::push_heap(a_heap_.begin(), a_heap_.end(), a_cmp_);

            // push to ARA* heap if fscore different
            //if (states_[state_index].fscore(a_cmp_.eps)
            //        != states_[state_index].fscore(ara_cmp_.eps))
            if (ara_cmp_.eps < 1.0)
            {
                ara_heap_.push_back(state_index);
                std::push_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            }

            // push state's hash to visited map
            visited_.insert({states_[state_index].hash, state_index});
        }

        /** returns {state_index, eps with which state was selected} */
        std::tuple<size_t, FloatT> pop_state()
        {
            size_t state_index;
            FloatT eps;
            if (!ara_heap_.empty() && num_steps_%2 == 1)
            {
                std::pop_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
                state_index = ara_heap_.back();
                ara_heap_.pop_back();
                eps = ara_cmp_.eps;
            }
            else
            {
                std::pop_heap(a_heap_.begin(), a_heap_.end(), a_cmp_);
                state_index = a_heap_.back();
                a_heap_.pop_back();
                eps = a_cmp_.eps;
            }
            return {state_index, eps};
        }

        void push_solution(size_t state_index, FloatT eps)
        {
            states_[state_index].is_expanded = true;
            solutions_.push_back({ state_index, eps, time_since_start() });

            // sort solutions
            for (size_t i = solutions_.size()-1; i > 0; --i)
            {
                State& s1 = states_[solutions_[i-1].state_index];
                State& s2 = states_[solutions_[i].state_index];
                if (s1.g < s2.g)
                    std::swap(s1, s2);
            }
        }

        bool in_visited(const State& state) const
        {
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
                << "box=" << s.box << ", "
                << "leafs=";

            std::vector<NodeId> buf;
            find_node_ids(s, buf);
            for (size_t i = 0; i < g_.num_independent_sets(); ++i)
                if (buf[i] == -1) o << "? ";
                else o << buf[i] << ' ';
                
            o << "hash=" << s.hash << " }";
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
