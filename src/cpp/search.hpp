/**
 * \file search.hpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_SEARCH_HPP
#define VERITAS_SEARCH_HPP

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
    struct BaseState {
        BoxRef box;
        int indep_set;

        BaseState() : box(BoxRef::null_box()), indep_set(-1) {}
    };

    /** \private */
    struct MaxOutputState : public BaseState {
        FloatT g, h;
        MaxOutputState() : BaseState(), g(0.0), h(0.0) {}
    };

    template <typename Heuristic> class Search;

    struct MaxOutputHeuristic {
        using State = MaxOutputState;

        /** Fields `box` and `indep_set` of `out` must be set */
        bool update_heuristic(
                State& out,
                const Search<MaxOutputHeuristic>& search,
                const State& parent,
                const Graph::Vertex& merged_vertex) const;

        void print_state(std::ostream& strm, const State& s)
        {
            strm << "State g=" << s.g << ", h=" << s.h
                << ", f=" << open_score(s)
                << ", set=" << s.indep_set
                << std::endl;
        }
        
        FloatT open_score(const State& state) const
        { return state.g + state.h; }

        bool cmp_open_score(const State& a, const State& b) const
        { return open_score(a) < open_score(b); }

        FloatT focal_score(const State& state) const
        { return state.indep_set; } // deeper solution first
    };

    template <typename State>
    struct Solution {
        double time; 
        FloatT eps;
        State state;
    };

    struct Snapshot {
        double time = 0.0;
        size_t num_steps = 0;
        size_t num_solutions = 0;
        size_t num_open = 0;
        FloatT eps = 0.0;
        std::tuple<FloatT, FloatT, FloatT> bounds = {-FLOATT_INF, FLOATT_INF, FLOATT_INF}; // lo, up_a, up_ara
    };

    template <typename Heuristic>
    class Search {
        AddTree at_;
        Graph graph_;
        size_t mem_capacity_;
        time_point start_time_;

        friend Heuristic;
        using State = typename Heuristic::State;
        Heuristic heuristic_;

        std::vector<State> open_;
        std::vector<Solution<State>> solutions_;

        BlockStore<DomainPair> store_;
        mutable struct {
            Box box;
            std::vector<size_t> focal;
        } workspace_;

    public:

        // settings
        FloatT eps = 0.95;
        size_t max_focal_size = 100;
        bool debug = true;

        enum class StopReason {
            NO_REASON,
            NUM_SOLUTION_EXCEEDED,
            NUM_NEW_SOLUTION_EXCEEDED,
            OPTIMAL,
            UPPER_LT,
            LOWER_GT,
        };

        // TODO implement this, and step_for
        int stop_when_num_solutions_exceeds = 1;
        int stop_when_num_new_solutions_exceeds = 1;
        bool stop_when_optimal = true;
        FloatT stop_when_upper_less_than = -FLOATT_INF;
        FloatT stop_when_lower_greater_than = FLOATT_INF;

        StopReason stop_reason = StopReason::NO_REASON;

        // statistics
        size_t num_steps = 0;
        std::vector<Snapshot> snapshots;

    public:
        Search(const AddTree& at)
            : at_(at.neutralize_negative_leaf_values())
            , graph_(at_)
            , mem_capacity_(size_t(1024)*1024*1024)
            , start_time_{std::chrono::system_clock::now()}
        {
            init_();
        }

        bool step()
        {
            ++num_steps;

            if (open_.empty())
                return true;

            //State state = (num_steps%2 == 1)
            //    ? pop_from_focal_(eps)
            //    : pop_top_();
            //
            State state = pop_from_focal_();
            //heuristic_.print_state(std::cout, state);

            if (is_solution_(state))
            {
                size_t sol_index = push_solution_(state);
                const auto& sol = solutions_[sol_index];
                std::cout << "-- solution! " << sol_index
                    << ", " << num_steps
                    << ", f=" << heuristic_.open_score(sol.state)
                    << std::endl;
            }
            else
            {
                expand_(state);
            }

            return false;
        }

        bool steps(size_t num_steps)
        {
            bool done = false;
            //size_t num_sol = num_solutions();
            for (size_t i = 0; !done && i < num_steps; ++i)
            {
                done = step();

                // TODO stop when...
            }
            push_snapshot();
            return done;
        }

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

        size_t num_solutions() const { return solutions_.size(); }
        size_t num_open() const { return open_.size(); }

        /** lower, upper + base_score */
        std::tuple<FloatT, FloatT, FloatT> current_bounds() const
        {
            auto &&[lo, up, top] = current_bounds_wo_base_score();
            return {lo+at_.base_score, up+at_.base_score, top+at_.base_score};
        }

        const Solution<State>& get_solution(size_t solution_index) const
        { return solutions_.at(solution_index); }

        /**
         * Is `get_solution(0)` the optimal solution?
         * \return true when certainly optimal, false otherwise (= maybe optimal)
         */
        bool is_optimal() const
        {
            auto&&[lo, hi, top] = current_bounds_wo_base_score();
            return lo == hi;
        }

        void push_snapshot()
        {
            snapshots.push_back({
                time_since_start(),
                num_steps,
                num_solutions(),
                num_open(),
                eps,
                current_bounds()});
        }



    private:
        void init_()
        {
            State initial_state, dummy_parent;
            Graph::Vertex dummy_v { 0, BoxRef::null_box(), 0.0 };
            heuristic_.update_heuristic(initial_state, *this, dummy_parent, dummy_v);
            push_(std::move(initial_state));
        }

        bool is_solution_(const State& state)
        {
            return state.indep_set+1 == static_cast<int>(graph_.num_independent_sets());
        }

        /** \return solution index */
        size_t push_solution_(const State& state)
        {
            solutions_.push_back({
                time_since_start(),
                eps,
                state,
            });

            auto cmp = [this](const State& a, const State& b) {
                return heuristic_.cmp_open_score(a, b); };

            // sort solutions
            size_t i = solutions_.size()-1;
            for (; i > 0; --i)
            {
                Solution<State>& sol1 = solutions_[i-1];
                Solution<State>& sol2 = solutions_[i];
                if (cmp(sol1.state, sol2.state))
                    std::swap(sol1, sol2);
                else return i;
            }
            return 0;
        }

        void expand_(const State& state)
        {
            int next_indep_set = state.indep_set + 1;
            Graph::IndepSet set = graph_.get_vertices(next_indep_set);
            int num_vertices = static_cast<int>(set.size());
            for (int vertex = 0; vertex < num_vertices; ++vertex)
            {
                const Graph::Vertex& v = set[vertex];
                if (v.box.overlaps(state.box))
                {
                    combine_boxes(v.box, state.box, true, workspace_.box);
                    construct_and_push_states_(state, v);
                }
            }
        }

        void construct_and_push_states_(const State& parent, const Graph::Vertex& v)
        {
            auto push_workspace_box_fun = [this, parent, &v](Box& b){
                State new_state;
                new_state.indep_set = parent.indep_set + 1;
                new_state.box = BoxRef(store_.store(b, remaining_mem_capacity()));
                if (heuristic_.update_heuristic(new_state, *this, parent, v))
                    push_(std::move(new_state));
                else std::cout << " -> invalid heuristic, skipping" << std::endl;
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

        void push_(State&& state)
        {
            //size_t state_index = push_state_(std::move(state));
            auto cmp = [this](const State& a, const State& b) {
                return heuristic_.cmp_open_score(a, b); };
            push_to_heap_(open_, std::move(state), cmp);
            //return state_index;
        }

        State pop_top_()
        {
            auto cmp = [this](const State& a, const State& b) {
                return heuristic_.cmp_open_score(a, b); };
            return pop_from_heap_(open_, cmp);
        }

        State pop_from_focal_()
        {
            if (eps == 1.0)
                return pop_top_();
            if (max_focal_size <= 1)
                return pop_top_();

            auto cmp_i = [this](size_t a, size_t b) {
                return heuristic_.cmp_open_score(open_[a], open_[b]); };
            auto cmp_s = [this](const State& a, const State& b) {
                return heuristic_.cmp_open_score(a, b); };

            FloatT fmin = eps*heuristic_.open_score(open_.front());
            FloatT foc_best = -FLOATT_INF;
            FloatT i_best = 0;
            size_t focal_size = 0;

            workspace_.focal.clear();
            workspace_.focal.push_back(0);
            while (!workspace_.focal.empty())
            {
                size_t i = pop_from_heap_(workspace_.focal, cmp_i);
                const State& s = open_[i];

                FloatT foc = heuristic_.focal_score(s);
                if (foc > foc_best)
                {
                    foc_best = foc;
                    i_best = i;
                }

                if (++focal_size >= max_focal_size)
                    break;

                //std::cout << "  i=" << i << ", " << workspace_.focal.size() << ": ";
                //heuristic_.print_state(std::cout, s);

                if (2*i+1 < open_.size() && heuristic_.open_score(open_[2*i+1]) >= fmin)
                    push_to_heap_(workspace_.focal, 2*i+1, cmp_i);
                if (2*i+2 < open_.size() && heuristic_.open_score(open_[2*i+2]) >= fmin)
                    push_to_heap_(workspace_.focal, 2*i+2, cmp_i);
            }

            std::cout << "BEST CHOICE " << i_best << ", focal_score " << foc_best
                << ", f=" << heuristic_.open_score(open_[i_best])
                << " (vs " << heuristic_.open_score(open_.front()) << ")" << std::endl;
            return pop_index_heap_(open_, i_best, cmp_s);
        }

        template <typename T, typename CmpT>
        void push_to_heap_(std::vector<T>& heap, T&& s, const CmpT& cmp)
        {
            heap.push_back(std::move(s));
            std::push_heap(heap.begin(), heap.end(), cmp);
        }

        template <typename T, typename CmpT>
        T pop_from_heap_(std::vector<T>& heap, const CmpT& cmp)
        {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            T s = heap.back();
            heap.pop_back();
            return s;
        }

        template <typename T, typename CmpT>
        T pop_index_heap_(std::vector<T>& heap, size_t index, const CmpT& cmp)
        {
            if (index == 0)
                return pop_from_heap_(heap, cmp);

            std::swap(heap.back(), heap[index]);
            T s = heap.back();
            heap.pop_back();

            //std::cout << "BEFORE\n";
            //print_heap(heap, 0);

            // heapify up
            for (size_t i = index; i != 0;)
            {
                size_t parent = (i-1)/2;
                if (cmp(heap[i], heap[parent])) // parent larger than i
                    break; // heap prop satisfied
                std::cout << "heapify up " << i << " <-> " << parent << std::endl;
                std::swap(heap[i], heap[parent]);
                i = parent;
            }

            // heapify down:
            // https://courses.cs.duke.edu/spring05/cps130/lectures/littman.lectures/lect08/node16.html
            for (size_t i = index;;)
            {
                size_t left = i*2 + 1;
                size_t right = i*2 + 2;
                bool has_left = left < heap.size();
                bool has_right = right < heap.size();

                if ((!has_left || cmp(heap[left], heap[i]))
                        && (!has_right || cmp(heap[right], heap[i])))
                    break;

                size_t larger = left;
                if (has_right && cmp(heap[left], heap[right]))
                    larger = right;

                //std::cout << "heapfy down " << i << " <-> " << larger;
                //std::cout << " fscores " << heuristic_.open_score(heap[i])
                //    << ", " << heuristic_.open_score(heap[larger])
                //    << " (" << heuristic_.open_score(heap[larger==left ? right : left]) << ")" << std::endl;

                std::swap(heap[larger], heap[i]);
                i = larger;
            }

            //std::cout << "AFTER\n";
            //print_heap(heap, 0);

            if (debug && !std::is_heap(heap.begin(), heap.end(), cmp))
            {
                print_heap(heap);
                auto until = std::is_heap_until(heap.begin(), heap.end(), cmp);
                std::cout << "heap until " << (until-heap.begin()) << ", "
                    << heuristic_.open_score(*until) << std::endl;
                throw std::runtime_error("whoops not a heap");
            }
            return s;
        }


        void print_heap(const std::vector<State>& v, size_t i=0, size_t depth=0)
        {
            if (i >= v.size())
                return;
            for (size_t j = 0 ; j < depth; ++j)
                std::cout << "  ";
            std::cout << i << ": " << heuristic_.open_score(v[i]) << std::endl;

            print_heap(v, i*2 + 1, depth+1);
            print_heap(v, i*2 + 2, depth+1);
        }

        /** lower, upper, top of open */
        std::tuple<FloatT, FloatT, FloatT> current_bounds_wo_base_score() const
        {
            FloatT lo = -FLOATT_INF, up = FLOATT_INF, top = FLOATT_INF;
            if (open_.size() > 0)
            {
                top = heuristic_.open_score(open_.front());
                up = top;
            }
            if (num_solutions() > 0)
            {
                lo = heuristic_.open_score(solutions_[0].state); // best solution so far, sols are sorted
                if (open_.size() == 0 || (up < lo))
                    up = lo;
            }
            return {lo, up, top};
        }


    }; // class Search

    inline bool
    MaxOutputHeuristic::update_heuristic(
            State& out,
            const Search<MaxOutputHeuristic>& search,
            const State& parent,
            const Graph::Vertex& merged_vertex) const
    {
        FloatT g = parent.g + merged_vertex.output;
        FloatT h = search.graph_.basic_remaining_upbound(out.indep_set+1,
                out.box);

        if (!std::isinf(h))
        {
            out.g = g;
            out.h = h;
            return true;
        }
        else return false;
    }

} // namespace veritas

#endif // VERITAS_SEARCH_HPP
