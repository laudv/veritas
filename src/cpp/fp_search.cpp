/**
 * \file fp_search.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "fp_search.hpp"
#include "block_store.hpp"
#include "box.hpp"
#include <chrono>

namespace veritas {

std::ostream& operator<<(std::ostream& strm, StopReason r)
{
#define VER_STOP_REASON_CASE(name) case StopReason::name: \
    strm << #name; \
    break;

    switch (r) {
        VER_STOP_REASON_CASE(NONE)
        VER_STOP_REASON_CASE(NO_MORE_OPEN)
        VER_STOP_REASON_CASE(NUM_SOLUTIONS_EXCEEDED)
        VER_STOP_REASON_CASE(NUM_NEW_SOLUTIONS_EXCEEDED)
        VER_STOP_REASON_CASE(OPTIMAL)
        VER_STOP_REASON_CASE(UPPER_LT)
        VER_STOP_REASON_CASE(LOWER_GT)
    }

    return strm;
#undef VER_STOP_REASON_CASE
}

template <typename Heuristic>
class SearchImpl;

using MaxCmp = std::greater<FloatT>;
using MinCmp = std::less<FloatT>;

/**
 * A state for ensemble output optimization.
 */
struct OutputState {
    /**
     * Sum of uniquely selected leaf values so far.
     */
    FloatT gscore;

    /**
     * Overestimate (maximization) or underestimate (minimization) of output
     * that can still be added to g by trees for which multiple leaves are
     * still reachable.
     */
    FloatT hscore;

    /**
     * Cached focal score, computed by heuristic computation.
     */
    FloatT fscore;

    /**
     * Which tree do we merge into this state next? This is determined by
     * the heuristic computation.
     */
    int next_tree;

    /**
     * Scoring function for this state for the open list.
     */
    FloatT open_score() const {
        return gscore + hscore;
    }

    /**
     * Scoring function for this state for the focal list.
     */
    FloatT focal_score() const {
        return fscore;
    }
};

template <typename OpenCmp, typename FocalCmp>
struct BasicOutputHeuristic {
    using State = OutputState;

    OpenCmp open_cmp;
    FocalCmp focal_cmp;

    BasicOutputHeuristic() : open_cmp{}, focal_cmp{} {}

    void update_scores(const SearchImpl<BasicOutputHeuristic>& s, State& out) {
        FloatT h = 0.0;
        s.workspace_.leafiter2.setup_flatbox(state.box); // do once
        for (size_t tree_index = state.indep_set + 1;
                tree_index < s.at_.size(); ++tree_index)
        {
            FloatT max = -FLOATT_INF;
            const Tree& t = s.at_[tree_index];
            s.workspace_.leafiter2.setup_tree(t);
            NodeId leaf_id = -1;
            while ((leaf_id = s.workspace_.leafiter2.next()) != -1)
            {
                if (s.node_box_[tree_index][leaf_id].is_invalid_box())
                    continue;
                max = std::max(t[leaf_id].leaf_value(), max);
            }
            h += max;
        }
    }
};

template <typename OpenCmp, typename FocalCmp>
struct CountingOutputHeuristic
    : public BasicOutputHeuristic<OpenCmp, FocalCmp> {

};


template <typename Heuristic>
class SearchImpl : public Search {
private:
    using State = typename Heuristic::State;

    std::shared_ptr<Heuristic> h_;
    std::vector<State> open_;

    BlockStore<IntervalPairFp> store_;

public:
    SearchImpl(const AddTree& at, std::shared_ptr<Heuristic> h)
        : Search(at)
        , h_{std::move(h)}
        , open_{}
        , store_{}
    {}

public: // virtual Search methods

    StopReason step() override {
        return step_();
    }

    StopReason steps(size_t num_steps) override {
        return StopReason::NONE;
    }

    StopReason step_for(double num_seconds, size_t num_steps) override {
        return StopReason::NONE;
    }

    size_t num_open() const override {
        return 0;

    }

    bool is_optimal() const override {
        return false;

    }

    size_t num_solutions() const override {
        return 0;
    }

    //const Solution& get_solution(size_t solution_index) const override;

    void set_mem_capacity(size_t bytes) override {

    }

    size_t remaining_mem_capacity() const override {
        return 0;
    }

    size_t used_mem_size() const override {
        return 0;
    }
    
    Bounds current_bounds() const override {
        return {0.0, 0.0};
    }

private:
    mutable struct {
        ///** \private */ bool reject_flag;
        ///** \private */ Box box;
        ///** \private */ std::vector<IntervalFp> flatbox;
        ///** \private */ std::vector<IntervalFp> flatbox_frames;
        ///** \private */ std::vector<size_t> flatbox_offset;
        ///** \private */ std::vector<size_t> flatbox_caret;
        /** \private */ std::vector<size_t> focal;
        ///** \private */ LeafIter leafiter1; // expand_
        ///** \private */ LeafIter leafiter2; // heurstic computation
    } workspace_;

    StopReason step_() {
        ++stats.num_steps;

        if (open_.empty())
            return StopReason::NO_MORE_OPEN;

        State state = pop_from_focal_();

        // TODO expand

        return StopReason::NONE;
    }

    State pop_from_open_()
    {
        auto cmp = [this](const State& a, const State& b) {
            return heuristic.cmp_open_score(b, a); // (!) reverse: max-heap with less-than cmp
        };
        return pop_from_heap_(open_, cmp);
    }

    // J. Pearl and J. H. Kim, "Studies in Semi-Admissible Heuristics," in
    // IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.
    // PAMI-4, no. 4, pp. 392-399, July 1982, doi:
    // 10.1109/TPAMI.1982.4767270.
    State pop_from_focal_() {
        if (settings.focal_eps == 1.0)
            return pop_from_open_();
        if (settings.max_focal_size <= 1)
            return pop_from_open_();

        // reverse order of a and b, heap functions require less-than comparision
        auto cmp_i = [this](size_t a, size_t b) {
            return heuristic.cmp_open_score(open_[b], open_[a]); };
        auto cmp_s = [this](const State& a, const State& b) {
            return heuristic.cmp_open_score(b, a); };

        FloatT oscore = heuristic.open_score(open_.front());
        FloatT orelax = heuristic.relax_open_score(oscore, eps);
        FloatT i_best = 0;
        size_t focal_size = 0;

        workspace_.focal.clear();
        workspace_.focal.push_back(0);
        while (!workspace_.focal.empty())
        {
            size_t i = pop_from_heap_(workspace_.focal, cmp_i);
            const State& s = open_[i];

            //FloatT foc = heuristic.focal_score(s);
            //if (foc > foc_best)
            if (heuristic.cmp_focal_score(s, open_[i_best]))
            {
                i_best = i;
                //foc_best = foc;
            }

            if (++focal_size >= max_focal_size)
                break;

            //std::cout << num_steps << ": " << "ref_oscore=" << oscore << ", orelax=" << orelax
            //    <<", best=" << heuristic.open_score(open_[i_best]) << "/" << open_[i_best].indep_set << ": ";
            //heuristic.print_state(std::cout, s);

            if (2*i+1 < open_.size())
            {
                FloatT oscore1 = heuristic.open_score(open_[2*i+1]);
                if (heuristic.cmp_open_score(oscore1, orelax))
                    push_to_heap_(workspace_.focal, 2*i+1, cmp_i);
            }

            if (2*i+2 < open_.size())
            {
                FloatT oscore2 = heuristic.open_score(open_[2*i+2]);
                if (heuristic.cmp_open_score(oscore2, orelax))
                    push_to_heap_(workspace_.focal, 2*i+2, cmp_i);
            }
        }

        sum_focal_size_ += focal_size;

        //std::cout << "BEST CHOICE " << i_best << ", focal_score " << foc_best
        //    << ", f=" << heuristic.open_score(open_[i_best])
        //    << " (vs " << heuristic.open_score(open_.front()) << ")" << std::endl;
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
        //if constexpr (std::is_same<T, State>::value)
        //    std::cout << "first " << heuristic.open_score(s)
        //        << ", second " << heuristic.open_score(heap.front()) << std::endl;
        return s;
    }

}; // class SearchImpl






// Constructor methods
std::shared_ptr<Search>
Search::max_output(const AddTree& at) {
    return std::shared_ptr<Search>(
        new SearchImpl<BasicOutputHeuristic<MaxCmp, MaxCmp>>(at));
}

std::shared_ptr<Search>
Search::min_output(const AddTree& at) {
    return std::shared_ptr<Search>(
        new SearchImpl<BasicOutputHeuristic<MinCmp, MaxCmp>>(at));
}


// Helper methods in abstract class Search
double Search::time_since_start() const
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            now-start_time_).count() * 1e-6;
}

} // namespace veritas
