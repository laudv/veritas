/**
 * \file fp_search.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "fp_search.hpp"
#include "addtree.hpp"
#include "basics.hpp"
#include "block_store.hpp"
#include "box.hpp"
#include "heuristics.hpp"
#include "leafiter.hpp"
#include "tree.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

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
        VER_STOP_REASON_CASE(ATLEAST_BOUND_BETTER_THAN)
        VER_STOP_REASON_CASE(OUT_OF_TIME)
        VER_STOP_REASON_CASE(OUT_OF_MEMORY)
    }

    return strm;
#undef VER_STOP_REASON_CASE
}

// heuristic policies (OutputState, *IsWorse, OutputHeuristic and its
// subclasses) live in heuristics.hpp; this file is the search engine that
// drives them.

template <typename State>
struct SolutionImpl {
    State state;
    double time;

    SolutionImpl(State&& s, double t) : state(std::move(s)), time(t) {}
};

template <typename Heuristic>
class SearchImpl : public Search {
private:
    using State = typename Heuristic::State;
    using OpenIsWorseT = typename Heuristic::OpenIsWorseT;
    using FocalIsWorseT = typename Heuristic::FocalIsWorseT;

    std::vector<State> open_;
    std::vector<size_t> focal_;
    std::vector<SolutionImpl<State>> solutions_;
    mutable LeafIter<TreeFp> leafiter_;
    BoxFp::BufT boxbuf_; // buffer to construct box of new state

    bool is_oom_;

public:
    std::shared_ptr<Heuristic> heuristic;

    SearchImpl(const Config& config, std::shared_ptr<Heuristic> h,
            const AddTree& at, const FlatBox& prune_box)
        : Search(config, at, prune_box)
        , open_{}
        , focal_{}
        , solutions_{}
        , leafiter_{}
        , boxbuf_{}
        , is_oom_{false}
        , heuristic{std::move(h)} { init_(); }

  public: // virtual Search methods

    StopReason step() override {
        return step_(); // call the non-virtual step_
    }

    StopReason steps(size_t num_steps) override {
        size_t num_sols_at_start = num_solutions();
        return steps_(num_steps, num_sols_at_start);
    }

    StopReason step_for(double num_seconds, size_t num_steps) override {
        double start = time_since_start();
        size_t num_sols_at_start = num_solutions();
        StopReason reason = StopReason::NONE;

        while (reason == StopReason::NONE) {
            reason = steps_(num_steps, num_sols_at_start);
            double dur = time_since_start() - start;
            if (dur >= num_seconds)
                return StopReason::OUT_OF_TIME;
        }

        return reason;
    }

    size_t num_open() const override {
        return open_.size();
    }

    bool is_optimal() const override {
        return !solutions_.empty() &&
               (open_.empty() ||
                heuristic->open_isworse(open_[0], solutions_[0].state));
    }

    size_t num_solutions() const override {
        return solutions_.size();
    }

    Solution get_solution(size_t solution_index) const override {
        const auto& sol = solutions_.at(solution_index);

        Box::BufT buf;

        // Start from the prune_box...
        int sz = static_cast<int>(prune_box_.size());
        for (NodeId fid = 0; fid < sz; ++fid) {
            const auto& ival = prune_box_[fid];
            if (!ival.is_everything()) {
                buf.emplace_back(fid, fpmap_.itransform(fid, ival));
            }
        }

        // ... and then combine with the state's box
        Box box{buf};
        for (auto&&[fid, ival] : sol.state.box)
            box.refine_box(fid, fpmap_.itransform(fid, ival));

        return {
            std::move(buf),
            sol.state.open_score(),
            sol.time
        };
    }

    std::vector<NodeId> get_solution_nodes(size_t solution_index) const override {
        const auto& sol = solutions_.at(solution_index);
        std::vector<NodeId> nodes;
        for (const TreeFp& tree : atfp_) {
            leafiter_.setup(tree, sol.state.box, prune_box_);
            NodeId leaf_id = leafiter_.next();
            if (leafiter_.next() != -1)
                throw std::runtime_error("no unique output for box");
            nodes.push_back(leaf_id);
        }
        return nodes;
    }

    Bounds current_bounds() const override {
        Bounds b(heuristic->open_isworse);
        if (open_.size() > 0) {
            b.top_of_open = open_.front().open_score();
            b.best = b.top_of_open;
        }
        if (num_solutions() > 0) {
            // best solution so far, sols are sorted
            b.atleast = solutions_[0].state.open_score();
            if (is_optimal())
                b.best = b.atleast;
        }
        return b;
    }

private:
    size_t get_remaining_unallocated_memory_() const {
#if VERITAS_SANITY_CHECKS
        if (config.max_memory < store_.get_mem_size())
            throw std::runtime_error("max memory underflow");
#endif
        return config.max_memory - store_.get_mem_size();
    }

    void init_() {
        State initial_state;
        bool success = heuristic->update_scores(atfp_, prune_box_,
                                               initial_state);
        push_to_open_if_valid_(std::move(initial_state), success);
    }

    bool is_solution_(const State& state) {
        // there is no more tree to add
        return state.next_tree == -1;
    }

    StopReason step_() {
        if (is_oom_)
            throw std::runtime_error(
                "Cannot continue, previous StopReason was OUT_OF_MEMORY.");
        if (open_.empty())
            return StopReason::NO_MORE_OPEN;

        ++stats.num_steps;

        State state = pop_from_focal_();

        if (is_solution_(state)) {
            /*size_t sol =*/ push_solution_(std::move(state));
            //const auto& solsol = solutions_[sol];

        } else {
            try {
                expand_(state);
            } catch (const BlockStoreOOM&) {
                leafiter_.clear();

                /*
                 * This leaves Veritas in an invalid state because some state
                 * expansions might not have been added to the open list. We
                 * can't really recover from this, so the next call to step_
                 * should lead to an exception.
                 */
                is_oom_ = true;

                return StopReason::OUT_OF_MEMORY;
            }
        }

        if (config.stop_when_optimal && is_optimal()) {
            return StopReason::OPTIMAL;
        }
        if (num_solutions() >= config.stop_when_num_solutions_exceeds) {
            return StopReason::NUM_SOLUTIONS_EXCEEDED;
        }
        if (num_solutions() > 0 && heuristic->open_isworse(
                config.stop_when_atleast_bound_better_than,
                solutions_[0].state.open_score())) {
            return StopReason::ATLEAST_BOUND_BETTER_THAN;
        }

        return StopReason::NONE;
    }

    StopReason steps_(size_t num_steps, size_t num_sols_at_start) {
        StopReason reason = StopReason::NONE;
        for (size_t i = 0; i < num_steps; ++i) {
            reason = step_();

            if (reason != StopReason::NONE)
                return reason;

            if (num_sols_at_start +
                    config.stop_when_num_new_solutions_exceeds
                    <= num_solutions()) {
                return StopReason::NUM_NEW_SOLUTIONS_EXCEEDED;
            }
        }
        return reason;
    }

    void expand_(const State& state) {
        //std::cout << "EXPANDING o=" << state.open_score()
        //          << ", f=" << state.focal_score()
        //          << ", next=" << state.next_tree
        //          << ", t=" << time_since_start() << "s"
        //          << ", m=" << (get_used_memory()/1024.0/1024.0) << "mb"
        //          << std::endl;
        const TreeFp& t = atfp_[state.next_tree];
        leafiter_.setup(t, state.box, prune_box_);

        for (NodeId leaf_id = leafiter_.next(); leaf_id != -1;
             leaf_id = leafiter_.next()) {
            construct_and_push_state_(state, t, leaf_id);
        }
    }

    void construct_and_push_state_(const State& parent, const TreeFp& t,
                                   NodeId leaf_id) {
        // Construct the box for the new state
        boxbuf_.clear();
        std::copy(parent.box.begin(), parent.box.end(),
                  std::back_inserter(boxbuf_));

        BoxFp box{boxbuf_};
        for (NodeId par_id = leaf_id; !t.is_root(par_id);) {
            bool is_left = t.is_left_child(par_id);
            par_id = t.parent(par_id);
            box.refine_box(t.get_split(par_id), is_left);
        }

        auto ref = store_.store(boxbuf_, get_remaining_unallocated_memory_());

        State new_state;
        new_state.box = BoxRefFp(ref.begin, ref.end);
        bool success = heuristic->update_scores(atfp_, prune_box_, new_state);

        push_to_open_if_valid_(std::move(new_state), success);
    }

    State pop_from_open_() {
        return pop_from_heap_(open_, heuristic->open_isworse);
    }

    void push_to_open_(State&& state) {
        push_to_heap_(open_, std::move(state), heuristic->open_isworse);
    }

    void push_to_open_if_valid_(State&& state, bool update_score_success) {
        if (!state.is_valid_state()) {
            std::cout << "Warning: new state invalid\n";
        } else if (!update_score_success) {
            ++stats.num_update_scores_fails;
        } else if (heuristic->open_isworse(
                   state.open_score(),
                   config.ignore_state_when_worse_than)) {
            ++stats.num_states_ignored;
        } else {
            push_to_open_(std::move(state));
        }
    }


    // J. Pearl and J. H. Kim, "Studies in Semi-Admissible Heuristics," in
    // IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.
    // PAMI-4, no. 4, pp. 392-399, July 1982, doi:
    // 10.1109/TPAMI.1982.4767270.
    State pop_from_focal_() {
        if (config.focal_eps == 1.0)
            return pop_from_open_();
        if (config.max_focal_size <= 1)
            return pop_from_open_();

        auto focal_cmp = [this](size_t i, size_t j) -> bool {
            const State& a = open_[i];
            const State& b = open_[j];

            // We _do_ sort by open_score
            // The point is to visit the states in `open_` with open_scores that
            // are better than `orelax`. Out of the visited states, we pick the
            // one state with the best focal_score.
            return heuristic->open_isworse(a.open_score(), b.open_score());
        };

        FloatT oscore = open_.front().open_score();
        FloatT orelax = heuristic->open_isworse.relax_open_score(
            oscore, config.focal_eps);
        size_t i_best = 0;

        focal_.clear();
        focal_.push_back(0);
        while (!focal_.empty()) {
            size_t i = pop_from_heap_(focal_, focal_cmp);
            const State& state = open_[i];

            // keep the old index if equal, earlier ones will have better
            // open_scores
            if (heuristic->focal_isworse(open_[i_best].focal_score(),
                                         state.focal_score())) {
                i_best = i;
            }

            if (focal_.size() >= config.max_focal_size)
                break;

            for (size_t child : {2*i+1, 2*i+2}) {
                if (child >= open_.size()) continue;

                FloatT oscore_child = open_[child].open_score();

                // Only push when the child state's open_score is better than
                // the relaxed score `orelax`
                if (heuristic->open_isworse(orelax, oscore_child))
                    push_to_heap_(focal_, std::move(child), focal_cmp);
            }
        }

        //sum_focal_size_ += focal_size;

        //std::cout << "BEST CHOICE " << i_best << ", focal_score "
        //    << open_[i_best].focal_score()
        //    << ", f=" << open_[i_best].open_score()
        //    << " (vs " << open_.front().open_score() << ")" << std::endl;
        return pop_index_heap_(open_, i_best, heuristic->open_isworse);
    }

    template <typename T, typename CmpT>
    void push_to_heap_(std::vector<T>& heap, T&& s, const CmpT& cmp) {
        heap.push_back(std::move(s));
        std::push_heap(heap.begin(), heap.end(), cmp);
    }

    template <typename T, typename CmpT>
    T pop_from_heap_(std::vector<T>& heap, const CmpT& cmp) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        T s = heap.back();
        heap.pop_back();
        return s;
    }

    /**
     * max-heap with less-than comparison, i.e.,
     * cmp(a, b) == True <=> a must be lower in the heap than b
     */
    template <typename T, typename Cmp>
    T pop_index_heap_(std::vector<T>& heap, size_t index, Cmp cmp) {
        if (index == 0)
            return pop_from_heap_(heap, cmp);

        std::swap(heap.back(), heap[index]);
        T s = heap.back();
        heap.pop_back();

        // heapify up
        for (size_t i = index; i != 0;) {
            size_t parent = (i-1)/2;
            if (!cmp(heap[parent], heap[i])) // parent lteq than i
                break; // heap prop satisfied
            //std::cout << "\033[91m" << "heapify up " << i << " <-> " << parent
            //          << ", " << heap[i].open_score()
            //          << ", " << heap[parent].open_score()
            //          << "\033[0m" << std::endl;
            std::swap(heap[i], heap[parent]);
            i = parent;
        }

        // heapify down:
        // https://courses.cs.duke.edu/spring05/cps130/lectures/littman.lectures/lect08/node16.html
        for (size_t i = index;;) {
            size_t left = i*2 + 1;
            size_t right = i*2 + 2;
            bool has_left = left < heap.size();
            bool has_right = right < heap.size();

            if ((!has_left || !cmp(heap[i], heap[left]))
                    && (!has_right || !cmp(heap[i], heap[right])))
                break;

            size_t larger = left;
            if (has_right && !cmp(heap[right], heap[left]))
                larger = right;

            //std::cout << "\033[92m" << "heapfy down " << i << " <-> " << larger;
            //std::cout << " fscores " << heap[i].open_score()
            //    << ", " << heap[larger].open_score()
            //    << " (" << heap[larger==left ? right : left].open_score() << ")"
            //      << "\033[0m" << std::endl;
            std::swap(heap[larger], heap[i]);
            i = larger;
        }
        
        return s;
    }

    /** \return solution index */
    size_t push_solution_(State&& state) {
        solutions_.emplace_back(
            std::move(state),
            time_since_start()
        );

        const State& sol_state = solutions_.back().state;
        heuristic->notify_new_solution(atfp_, prune_box_, sol_state);

        //std::sort(solutions_.begin(), solutions_.end(),
        //        [this](const auto& s0, const auto& s1) {
        //            return heuristic->open_isworse(s0.state, s1.state);
        //        });

        // sort solutions
        size_t i = solutions_.size()-1;
        for (; i > 0; --i) {
            auto& sol1 = solutions_[i-1]; // should have better score
            auto& sol2 = solutions_[i];
            if (heuristic->open_isworse(sol1.state, sol2.state)) {
                std::swap(sol1, sol2);
            } else {
                return i;
            }
        }
        return 0;
    }

}; // class SearchImpl






template <typename H>
static void config_set_defaults(Config& c) {
    using OpenIsWorse = typename H::OpenIsWorseT;
    using OrdLim = OrdLimit<FloatT, OpenIsWorse>;

    OpenIsWorse cmp;
    c.ignore_state_when_worse_than = OrdLim::worst(cmp);
    c.stop_when_atleast_bound_better_than = OrdLim::best(cmp);
    c.multi_ignore_state_when_class0_worse_than = OrdLim::worst(cmp);
}

Config::Config(HeuristicType h)
    : heuristic(h)
    , ignore_state_when_worse_than(0.0)
    , stop_when_atleast_bound_better_than(0.0)
    , multi_ignore_state_when_class0_worse_than(0.0) {

    switch (heuristic) {
    case HeuristicType::MAX_OUTPUT:
        config_set_defaults<MaxBasicOutputHeuristic>(*this);
        break;
    case HeuristicType::MIN_OUTPUT:
        config_set_defaults<MinBasicOutputHeuristic>(*this);
        break;
    case HeuristicType::MAX_COUNTING_OUTPUT:
        config_set_defaults<MaxCountingOutputHeuristic>(*this);
        break;
    case HeuristicType::MIN_COUNTING_OUTPUT:
        config_set_defaults<MinCountingOutputHeuristic>(*this);
        break;
    case HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF:
        config_set_defaults<MaxMaxMultiOutputHeuristic>(*this);
        break;
    case HeuristicType::MULTI_MAX_MIN_OUTPUT_DIFF:
        config_set_defaults<MaxMinMultiOutputHeuristic>(*this);
        break;
    case HeuristicType::MULTI_MIN_MAX_OUTPUT_DIFF:
        config_set_defaults<MinMaxMultiOutputHeuristic>(*this);
        break;
    default:
        throw std::runtime_error("invalid HeuristicType in config (init)");
    }
}

std::shared_ptr<Search>
Config::get_search(const AddTree& at, const FlatBox& prune_box) const {
    switch (heuristic) {
    case HeuristicType::MAX_OUTPUT: {
        auto h = std::make_shared<MaxBasicOutputHeuristic>();
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
                *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MIN_OUTPUT: {
        auto h = std::make_shared<MinBasicOutputHeuristic>();
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
                *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MAX_COUNTING_OUTPUT: {
        auto h = std::make_shared<MaxCountingOutputHeuristic>();
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
            *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MIN_COUNTING_OUTPUT: {
        auto h = std::make_shared<MinCountingOutputHeuristic>();
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
            *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF: {
        auto h = std::make_shared<MaxMaxMultiOutputHeuristic>(
            *this, at.num_leaf_values(), at.size());
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
            *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MULTI_MAX_MIN_OUTPUT_DIFF: {
        auto h = std::make_shared<MaxMinMultiOutputHeuristic>(
            *this, at.num_leaf_values(), at.size());
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
            *this, std::move(h), at, prune_box);
    }
    case HeuristicType::MULTI_MIN_MAX_OUTPUT_DIFF: {
        auto h = std::make_shared<MinMaxMultiOutputHeuristic>(
            *this, at.num_leaf_values(), at.size());
        return std::make_shared<SearchImpl<decltype(h)::element_type>>(
            *this, std::move(h), at, prune_box);
    }
    default:
        throw std::runtime_error("invalid HeuristicType in config (get_search)");
    }
}

std::shared_ptr<Search>
Config::reuse_heuristic(const Search& search, const FlatBox& prune_box) const {
    if (search.config.heuristic != heuristic)
        throw std::runtime_error("incompatible heuristic setting");
    if (heuristic != HeuristicType::MAX_COUNTING_OUTPUT &&
        heuristic != HeuristicType::MIN_COUNTING_OUTPUT) {
    }

    switch (heuristic) {
    case HeuristicType::MAX_COUNTING_OUTPUT: {
        const auto& s = dynamic_cast<const SearchImpl<MaxCountingOutputHeuristic>&>(search);
        return std::make_shared<SearchImpl<MaxCountingOutputHeuristic>>(
            *this, s.heuristic, s.get_addtree(), prune_box);
    }
    case HeuristicType::MIN_COUNTING_OUTPUT: {
        const auto& s = dynamic_cast<const SearchImpl<MinCountingOutputHeuristic>&>(search);
        return std::make_shared<SearchImpl<MinCountingOutputHeuristic>>(
            *this, s.heuristic, s.get_addtree(), prune_box);
    }
    default:
        throw std::runtime_error("reuse_heuristic only available on counting heuristics");
    }
}

// Search constructor
Search::Search(const Config& config, const AddTree& at, const FlatBox& prune_box)
    : config{config}
    , stats{}
    , at_{at.neutralize_negative_leaf_values()}
    , atfp_{1, AddTreeType::REGR} // placeholder, replaced in constructor
    , fpmap_{}
    , start_time_{time_clock::now()}
    , store_{config.memory_min_block_size}
    , prune_box_{}
{
    if (config.max_memory < store_.get_mem_size()) {
        std::cout << config.max_memory << " vs " << store_.get_mem_size() << std::endl;
        throw std::runtime_error("max_memory too low");
    }

    fpmap_.add(at_);
    fpmap_.add(prune_box);
    fpmap_.finalize();
    atfp_ = fpmap_.transform(at_);

    // Convert prune_box to fixed precision, and push to leafiter
    int feat_id = 0;
    for (const Interval& ival : prune_box) {
        prune_box_.push_back(fpmap_.transform(feat_id++, ival));
    }
}


// Helper methods in abstract class Search
double Search::time_since_start() const {
    time_point now = time_clock::now();
    auto cnt = std::chrono::duration_cast<std::chrono::microseconds>(
            now-start_time_).count();
    return static_cast<double>(cnt) * 1e-6;
}

size_t Search::get_used_memory() const {
    return store_.get_used_mem_size();
}

const AddTree&
Search::get_addtree() const {
    return at_;
}

std::ostream& operator<<(std::ostream& s, const Bounds& bounds) {
    return s << "Bounds("
        << "atleast=" << bounds.atleast
        << ", best=" << bounds.best
        << ", top=" << bounds.top_of_open << ')';
}

std::ostream& operator<<(std::ostream& s, const Solution& sol) {
    BoxRef box{sol.box.begin(), sol.box.end()};
    return s << "Solution(" << box << ", output=" << sol.output << ')';
}

} // namespace veritas
