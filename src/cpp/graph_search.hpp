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
        FloatT bound = 0.0;
    };

    struct State {
        size_t parent; // index into GraphSearch::states_
        int indep_set; // index of last index_set added to this state
        int vertex; // index of last vertex added to this state
        FloatT g, h;
        BoxRef box;

        bool is_expanded;

        inline FloatT fscore(FloatT eps=1.0) const { return g + eps*h; }
    };

    std::ostream&
    operator<<(std::ostream& strm, const State& s)
    {
        return strm
            << "State {" << std::endl
            << "   - parent: " << s.parent << std::endl
            << "   - index_set, vertex: " << s.indep_set << ", " << s.vertex << std::endl
            << "   - g, h: " << s.g << ", " << s.h << std::endl
            << "   - box: " << s.box << std::endl
            << "   - expanded?: " << s.is_expanded << std::endl
            << "}";
    }

    struct StateCmp {
        const GraphSearch& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;

        inline bool operator()(const State& a, const State& b) const
        { return a.fscore(eps) < b.fscore(eps); }
    };

    class GraphSearch {
        Graph g_;

        BlockStore<DomainPair> store_;
        mutable struct { Box box; std::vector<int> indep_sets; } workspace_;

        std::vector<State> states_;
        std::vector<size_t> a_heap_, ara_heap_; // indexes into states_
        std::multimap<size_t, size_t> visited_; // hash -> state_index
        StateCmp a_cmp_, ara_cmp_;
        time_point start_time_;
        size_t num_steps_;
        
    public:
        friend StateCmp;

        GraphSearch(const AddTree& at)
            : g_(at)
            , a_cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.01}
            , num_steps_{0}
        {
            push_state({
                    0, // parent
                    0, // indep_set
                    0, // vertex
                    g_.get_vertices(0)[0].output, // g
                    0.0, // h
                    BoxRef::null_box(), // box
                    false // is_expanded
            }, 0);
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
            
            std::cout << "expanding " << states_[state_index] << std::endl;

            expand(state_index);

            return false;
        }

    private:
        void push_state(State&& state, size_t state_hash)
        {
            size_t state_index = states_.size();
            states_.push_back(std::move(state));

            a_heap_.push_back(state_index);
            std::push_heap(a_heap_.begin(), a_heap_.end(), a_cmp_);

            // push to ARA* heap if fscore different
            if (states_[state_index].fscore(a_cmp_.eps)
                    != states_[state_index].fscore(ara_cmp_.eps))
            {
                ara_heap_.push_back(state_index);
                std::push_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            }

            // push state's hash to visited map
            visited_.insert({state_hash, state_index});
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

        bool in_visited(const State& state, size_t state_hash) const
        {
            //size_t h = hash(state);
            auto r = visited_.equal_range(state_hash);

            for (auto it = r.first; it != r.second; ++it)
            {
                std::cout << "hash " << it->first << " -> " << it->second << " <=> " << state_hash;
                if (is_same_state(states_[it->second], state))
                {
                    std::cout << "YES";
                    return true;
                }
                std::cout << std::endl;
            }

            return false;
        }

        bool is_same_state(const State& s0, const State& s1) const
        {
            if (s0.indep_set != s1.indep_set) return false;
            if (s0.vertex != s1.vertex) return false;
            size_t s0_index = s0.parent;
            size_t s1_index = s1.parent;

            while (s0_index != 0 && s1_index != 0)
            {
                if (s0_index == s1_index)
                {
                    std::cout << "same because same index!" << std::endl;
                    return true;
                }
                const State& s0 = states_[s0_index];
                const State& s1 = states_[s1_index];
                if (s0.indep_set != s1.indep_set) return false;
                if (s0.vertex != s1.vertex) return false;
                s0_index = s0.parent;
                s1_index = s1.parent;
            }

            // TODO are boxes always the same? --> maybe that is a cheaper check

            return s0_index == s1_index; // if both ==0, then this is exactly the same state
        }

        bool expand(size_t state_index)
        {
            find_indep_sets_not_added(state_index); // fill workspace_.indep_sets

            states_[state_index].is_expanded = true;
            BoxRef state_box = states_[state_index].box;
            FloatT g = states_[state_index].g;

            for (int i : workspace_.indep_sets) // skip first `base_score` indep_set
            {
                Graph::IndepSet set = g_.get_vertices(i);
                int num_vertices = static_cast<int>(set.size());
                for (int j = 0; j < num_vertices; ++j) // filled by `find_indep_sets_not_added`
                {
                    const Graph::Vertex& v = set[j];
                    if (v.box.overlaps(state_box))
                    {
                        std::cout << "can add " << i << ", " << j << std::endl;
                        FloatT h = 0.0;
                        combine_boxes(v.box, state_box, true, workspace_.box);
                        BoxRef box = BoxRef(store_.store(workspace_.box, remaining_mem_capacity()));
                        workspace_.box.clear();
                        State new_state = {
                            state_index, // parent
                            i, /* indep_set */ j, /* vertex */
                            g+v.output,
                            h,
                            box,
                            false // is_expanded
                        };
                        size_t state_hash = hash(new_state);
                        if (!in_visited(new_state, state_hash))
                        {
                            std::cout << "not yet seen!" << std::endl;
                            push_state(std::move(new_state), state_hash);
                        }
                        else
                        {
                            std::cout << "already seen" << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "cannot add " << i << ", " << j << std::endl;
                    }
                }
            }

            return true;
        }

        void find_indep_sets_not_added(size_t state_index) const
        {
            workspace_.indep_sets.clear();
            for (size_t i=0; i<g_.num_independent_sets(); ++i)
                workspace_.indep_sets.push_back(i);

            while (state_index != 0)
            {
                const State& s = states_[state_index];
                workspace_.indep_sets[s.indep_set] = 0;
                state_index = s.parent;
            }

            workspace_.indep_sets.erase(
                    std::remove_if(
                        workspace_.indep_sets.begin(),
                        workspace_.indep_sets.end(),
                        [](int x) { return x == 0; }),
                    workspace_.indep_sets.end());
        }

    public:
        // generate a hash that is unique for each 
        size_t hash(const State& state) const // state possibly not yet added to states_
        {
            size_t h = hash(0, state.indep_set, state.vertex);

            size_t state_index = state.parent;
            while (state_index != 0)
            {
                const State& s = states_[state_index];
                h = hash(h, s.indep_set, s.vertex);
                state_index = s.parent;
            }
            return h;
        }

        size_t hash(size_t h, int indep_set, int vertex) const
        {
            size_t a = (67'280'421'310'721 * std::hash<int>()(indep_set))
                ^ (999'999'000'001 * std::hash<int>()(vertex));
            h = h ^ a;
            std::cout << "  hash round: " << indep_set << ", " << vertex << " => " << h << std::endl;
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
