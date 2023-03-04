/**
 * \file fp_search.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "fp_search.hpp"
#include "basics.hpp"
#include "block_store.hpp"
#include "box.hpp"
#include "leafiter.hpp"
#include <cmath> // isinf
#include <type_traits>

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
     * The input space restrictions of this state.
     */
    BoxRefFp box;

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

    /**
     * Does this state represent a valid path through the branches of the trees?
     * A state is invalid if the hscore is infinite, which happens when there is
     * at least one tree with not a remaining overlapping root-to-leaf branch.
     */
    bool is_valid_state() const {
        return !std::isinf(hscore);
    }

    /** Default constructor produces invalid state */
    OutputState()
        : gscore{}
        , hscore{Limits<FloatT>::max} // invalid
        , fscore{}
        , next_tree{-1}
        , box{}
    {}
};

struct LessIsWorse {
    using ValueT = FloatT;

    constexpr bool operator()(ValueT a, ValueT b) const {
        return std::less<ValueT>()(a, b);
    }
};

struct GreaterIsWorse {
    using ValueT = FloatT;

    constexpr bool operator()(ValueT a, ValueT b) const {
        return std::greater<ValueT>()(a, b);
    }
};

template <typename IsWorse> 
struct OutputStateOpenIsWorse {
    IsWorse cmp;
    OutputStateOpenIsWorse() : cmp(IsWorse()) {}

    operator IsWorse() const { return cmp; }

    bool operator()(FloatT a, FloatT b) const {
        return cmp(a, b);
    }

    bool operator()(const OutputState& a, const OutputState& b) const {
        return cmp(a.open_score(), b.open_score());
    }

    FloatT relax_open_score(FloatT open_score, FloatT eps) const {
        FloatT relaxation = std::abs(open_score) * (1.0 - eps);
        if constexpr (std::is_same_v<LessIsWorse, IsWorse>) {
            return open_score - relaxation;
        } else {
            static_assert(std::is_same_v<GreaterIsWorse, IsWorse>);
            return open_score + relaxation;
        } 
    }
};


// *IsWorse has an operator()(a, b) which returns true when a is 'worse' than b,
// i.e., b should be tried before a
template <typename OpenIsWorse, typename FocalIsWorse, typename GScoreF>
struct OutputHeuristic {
    using State = OutputState;
    using OpenIsWorseT = OpenIsWorse;
    using FocalIsWorseT = FocalIsWorse;

    OutputStateOpenIsWorse<OpenIsWorse> open_isworse;
    FocalIsWorse focal_isworse;
    LeafIter<TreeFp> leafiter;
    GScoreF gscore_f;

    OutputHeuristic(GScoreF f)
        : open_isworse{}, focal_isworse{}, leafiter{}, gscore_f{f} {}

    void update_scores(const AddTreeFp& at, const FlatBoxFp& prune_box,
                       State &state) {
        state.gscore = at.base_score;
        state.hscore = 0.0;
        state.next_tree = -1;
        FloatT best_of_best =
            OrdLimit<FloatT, OpenIsWorse>::worst(open_isworse);

        leafiter.setup_flatbox(state.box, prune_box);
        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index) {
            FloatT best = OrdLimit<FloatT, OpenIsWorse>::worst(open_isworse);;
            const auto t = at[tree_index];
            leafiter.setup_tree(t);
            int num_leaves = 0;
            NodeId leaf_id = leafiter.next();
            for (NodeId i = leaf_id; i != -1; i = leafiter.next()) {
                ++num_leaves;
                leaf_id = i; // store the last valid leaf_id (avoid -1)
                best = std::max(t.leaf_value(leaf_id), best, open_isworse);
            }
            if (leaf_id == -1)
                throw std::runtime_error("leaf_id == -1?");
            if (num_leaves > 1) { // multiple leaves reachable still
                state.hscore += best;
                if (open_isworse(best_of_best, best)) {
                    best_of_best = best;
                    state.next_tree = static_cast<int>(tree_index);
                }
            } else {
                gscore_f(at, state, tree_index, leaf_id);
            }
        }
    }
};

struct BasicOutputGScore {
    void operator()(const AddTreeFp& at, OutputState& state,
                    size_t tree_index, NodeId leaf_id) {
        state.gscore += at[tree_index].leaf_value(leaf_id);
        state.fscore += 1; // deeper solution first
    }
};

template <typename OpenIsWorse, typename FocalIsWorse>
struct BasicOutputHeuristic
    : public OutputHeuristic<OpenIsWorse, FocalIsWorse, BasicOutputGScore> {

    using BaseT = OutputHeuristic<OpenIsWorse, FocalIsWorse, BasicOutputGScore>;

    BasicOutputHeuristic() : BaseT(BasicOutputGScore()) {}
};

struct CountingOutputGScore {
    void operator()(const AddTreeFp& at, OutputState& state,
                    size_t tree_index, NodeId leaf_id) {
        state.gscore += at[tree_index].leaf_value(leaf_id);
        // TODO: Accumulate the count for this tree_id, leaf_id!
        // state.fscore += counts[tree_index][leaf_id];
    }
};

//template <typename OpenIsWorse, typename FocalIsWorse>
//struct CountingOutputHeuristic
//    : public OutputHeuristic<OpenIsWorse, FocalIsWorse, CountingOutputGScore> {
//
//    CountingOutputHeuristic() : CountingOutputHeuristic(BasicOutputGScore()) {}
//};

using MaxBasicOutputHeuristic = BasicOutputHeuristic<LessIsWorse, LessIsWorse>;
using MinBasicOutputHeuristic = BasicOutputHeuristic<GreaterIsWorse, LessIsWorse>;

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
    LeafIter<TreeFp> leafiter_;
    BoxFp::BufT boxbuf_; // buffer to construct box of new state

public:
    std::shared_ptr<Heuristic> heuristic;

    SearchImpl(const AddTree& at, const FlatBox& prune_box,
            std::shared_ptr<Heuristic> h)
        : Search(Settings(h->open_isworse), at, prune_box)
        , open_{}
        , focal_{}
        , solutions_{}
        , leafiter_{}
        , boxbuf_{}
        , heuristic{std::move(h)} { init_(); }

    template <typename H = Heuristic>
    SearchImpl(const AddTree& at, const FlatBox& prune_box)
        : SearchImpl(at, prune_box, std::make_shared<Heuristic>()) {}

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
        return max_memory_ - store_.get_mem_size();
    }

    void init_() {
        State initial_state;
        heuristic->update_scores(atfp_, prune_box_, initial_state);

        if (initial_state.is_valid_state())
            push_to_open_(std::move(initial_state));
        else
            std::cout << "Warning: initial_state invalid" << std::endl;
    }

    bool is_solution_(const State& state) {
        // there is no more tree to add
        return state.next_tree == -1;
    }

    StopReason step_() {
        if (open_.empty())
            return StopReason::NO_MORE_OPEN;

        ++stats.num_steps;

        State state = pop_from_focal_();

        if (is_solution_(state)) {
            //std::cout << "SOLUTION FOUND "
            //    << "open_score=" << state.open_score()
            //    << ", focal_score=" << state.focal_score() << '\n';
            push_solution_(std::move(state));
        } else {
            expand_(state);
        }

        if (settings.stop_when_optimal && is_optimal()) {
            return StopReason::OPTIMAL;
        }
        if (num_solutions() >= settings.stop_when_num_solutions_exceeds) {
            return StopReason::NUM_SOLUTIONS_EXCEEDED;
        }
        if (num_solutions() > 0 && heuristic->open_isworse(
                settings.stop_when_atleast_bound_better_than,
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
                    settings.stop_when_num_new_solutions_exceeds
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
        heuristic->update_scores(atfp_, prune_box_, new_state);

        if (!new_state.is_valid_state()) {
            std::cout << "Warning: new_state invalid\n";
        } else if (heuristic->open_isworse(
                       new_state.open_score(),
                       settings.ignore_state_when_worse_than)) {
            ++stats.num_states_ignored;
        } else {
            push_to_open_(std::move(new_state));
        }
    }

    State pop_from_open_() {
        return pop_from_heap_(open_, heuristic->open_isworse);
    }

    void push_to_open_(State&& state) {
        push_to_heap_(open_, std::move(state), heuristic->open_isworse);
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
            oscore, settings.focal_eps);
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

            if (focal_.size() >= settings.max_focal_size)
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

        //std::cout << "BEFORE\n";
        //print_heap_(heap, 0);

        // heapify up
        for (size_t i = index; i != 0;) {
            size_t parent = (i-1)/2;
            if (cmp(heap[i], heap[parent])) // parent larger than i
                break; // heap prop satisfied
            //std::cout << "heapify up " << i << " <-> " << parent << std::endl;
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

            if ((!has_left || cmp(heap[left], heap[i]))
                    && (!has_right || cmp(heap[right], heap[i])))
                break;

            size_t larger = left;
            if (has_right && cmp(heap[left], heap[right]))
                larger = right;

            //std::cout << "heapfy down " << i << " <-> " << larger;
            //std::cout << " fscores " << heuristic.open_score(heap[i])
            //    << ", " << heuristic.open_score(heap[larger])
            //    << " (" << heuristic.open_score(heap[larger==left ? right : left]) << ")" << std::endl;

            std::swap(heap[larger], heap[i]);
            i = larger;
        }

        //std::cout << "AFTER\n";
        //print_heap_(heap, 0);

        //if (debug && !std::is_heap(heap.begin(), heap.end(), cmp)) {
        //    print_heap_(heap);
        //    auto until = std::is_heap_until(heap.begin(), heap.end(), cmp);
        //    std::cout << "heap until " << (until-heap.begin()) << ", "
        //        << heuristic.open_score(*until) << std::endl;
        //    throw std::runtime_error("whoops not a heap");
        //}
        return s;
    }

    /** \return solution index */
    size_t push_solution_(State&& state) {
        solutions_.emplace_back(
            std::move(state),
            time_since_start()
        );

        // sort solutions
        size_t i = solutions_.size()-1;
        for (; i > 0; --i) {
            auto& sol1 = solutions_[i-1]; // should have better score
            auto& sol2 = solutions_[i];
            if (heuristic->open_isworse(sol1.state, sol2.state))
                std::swap(sol1, sol2);
            else return i;
        }
        return 0;
    }

}; // class SearchImpl






// Constructor methods
std::shared_ptr<Search>
Search::max_output(const AddTree& at, const FlatBox& prune_box) {
    return std::shared_ptr<Search>(
        new SearchImpl<MaxBasicOutputHeuristic>(at, prune_box));
}

std::shared_ptr<Search>
Search::min_output(const AddTree& at, const FlatBox& prune_box) {
    return std::shared_ptr<Search>(
        new SearchImpl<MinBasicOutputHeuristic>(at, prune_box));
}

// Search constructor
Search::Search(Settings s, const AddTree& at, const FlatBox& prune_box)
    : settings{s}
    , stats{}
    , at_{at.neutralize_negative_leaf_values()}
    , atfp_{}
    , fpmap_{}
    , start_time_{time_clock::now()}
    , max_memory_{settings.max_memory}
    , store_{}
    , prune_box_{}
{
    fpmap_.add(at_);
    fpmap_.add(prune_box);
    fpmap_.finalize();
    atfp_ = fpmap_.transform(at_);

    std::cout << "BASESCORE " << at.base_score << std::endl;
    std::cout << "BASESCORE " << at_.base_score << std::endl;
    std::cout << "BASESCORE " << atfp_.base_score << std::endl;

    // Convert prune_box to fixed precision, and push to leafiter
    int feat_id = 0;
    for (const Interval& ival : prune_box) {
        prune_box_.push_back(fpmap_.transform(feat_id++, ival));
    }
}


// Helper methods in abstract class Search
double Search::time_since_start() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            now-start_time_).count() * 1e-6;
}

void Search::set_max_memory(size_t bytes) { max_memory_ = bytes; }

size_t Search::get_max_memory() const { return max_memory_; }

size_t Search::get_used_memory() const {
    return store_.get_used_mem_size();
}

std::ostream& operator<<(std::ostream& s, const Bounds& bounds) {
    return s << "Bounds("
        << "atleast=" << bounds.atleast
        << ", best=" << bounds.best
        << ", top=" << bounds.top_of_open << ')';
}

std::ostream& operator<<(std::ostream& s, const Solution& sol) {
    BoxRef box{sol.box.begin(), sol.box.end()};
    return s << "Solution(box=" << box << ", output=" << sol.output << ')';
}

} // namespace veritas
