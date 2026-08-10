/**
 * \file heuristics.hpp
 *
 * Heuristic policies used by SearchImpl (fp_search.cpp): how a search state's
 * g/h/f scores are computed and updated as trees are merged in. This file
 * has no knowledge of the open/focal list mechanics of the search engine
 * itself -- see fp_search.cpp for that.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 */

#ifndef VERITAS_HEURISTICS_HPP
#define VERITAS_HEURISTICS_HPP

#include "basics.hpp"
#include "box.hpp"
#include "fp.hpp"
#include "fp_search.hpp" // Config
#include "leafiter.hpp"

#include <algorithm>
#include <cmath> // isinf
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace veritas {

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


template <typename OpenIsWorse, typename FocalIsWorse, typename Sub>
struct OutputHeuristic {
    using State = OutputState;

    // *IsWorse has an operator()(a, b) which returns true when a is 'worse'
    // than b, i.e., b should be tried before a
    using OpenIsWorseT = OpenIsWorse;
    using FocalIsWorseT = FocalIsWorse;

    OutputStateOpenIsWorse<OpenIsWorseT> open_isworse;
    FocalIsWorseT focal_isworse;
    LeafIter<TreeFp> leafiter;

    OutputHeuristic()
        : open_isworse{}, focal_isworse{}, leafiter{} {}

    bool update_scores(const AddTreeFp& at, const FlatBoxFp& prune_box,
                       State &state) {

        using OrdLim = OrdLimit<FloatT, OpenIsWorseT>;
        FloatT best_of_best = OrdLim::worst(open_isworse);

        static_cast<Sub *>(this)->initialize_state(at, state);

        leafiter.setup_flatbox(state.box, prune_box);
        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index) {
            FloatT best = OrdLim::worst(open_isworse);
            const auto& t = at[tree_index];
            leafiter.setup_tree(t);
            int num_leaves = 0;
            NodeId leaf_id = leafiter.next();
            for (NodeId i = leaf_id; i != -1; i = leafiter.next()) {
                ++num_leaves;
                leaf_id = i; // store the last valid leaf_id (avoid -1)
                best = std::max(t.leaf_value(leaf_id, 0), best, open_isworse);
            }
#ifdef VERITAS_SANITY_CHECKS
            if (leaf_id == -1)
                throw std::runtime_error("leaf_id == -1?");
#endif
            if (num_leaves > 1) { // multiple leaves reachable still
                state.hscore += best;
                if (open_isworse(best_of_best, best)) {
                    best_of_best = best;
                    state.next_tree = static_cast<int>(tree_index);
                }
            } else {
                static_cast<Sub *>(this)->update_gscore(
                        at, state, tree_index, leaf_id);
            }
        }
        return true; // true = success
    }

    /** Sub needs to implement this: */
    void initialize_gscore(const AddTreeFp & /*at*/,
                               OutputState & /*state*/, size_t /*tree_index*/,
                               NodeId /*leaf_id*/) {
        throw std::runtime_error("override in Sub!");
    };

    /** Sub needs to implement this: */
    void update_gscore(const AddTreeFp& /*at*/, OutputState& /*state*/,
                       size_t /*tree_index*/, NodeId /*leaf_id*/) {
        throw std::runtime_error("override in Sub!");
    };

    void update_hscore(const AddTreeFp& /*at*/ ) {
        throw std::runtime_error("override in Sub!");
    }

    /** Sub needs to implement this: */
    void notify_new_solution(const AddTreeFp& /*at*/,
                             const FlatBoxFp& /*prune_box*/,
                             const State& /*state*/) {
        throw std::runtime_error("override in Sub!");
    }
};

template <typename OpenIsWorse, typename FocalIsWorse>
struct BasicOutputHeuristic
    : public OutputHeuristic<OpenIsWorse, FocalIsWorse,
                             BasicOutputHeuristic<OpenIsWorse, FocalIsWorse>> {

    using OpenIsWorseT = OpenIsWorse;
    using FocalIsWorseT = FocalIsWorse;
    using SelfT = BasicOutputHeuristic<OpenIsWorse, FocalIsWorse>;
    using BaseT = OutputHeuristic<OpenIsWorse, FocalIsWorse, SelfT>;
    using State = typename BaseT::State;

    BasicOutputHeuristic() : BaseT() {}

    void initialize_state(const AddTreeFp& at, OutputState& state) {
        state.gscore = at.base_score(0);
        state.hscore = 0.0;
        state.next_tree = -1;
    }

    void update_gscore(const AddTreeFp& at, OutputState& state,
                       size_t tree_index, NodeId leaf_id) {
        state.gscore += at[tree_index].leaf_value(leaf_id, 0);
        state.fscore += 1; // deeper solution first
    };

    void notify_new_solution(const AddTreeFp& /*at*/,
                             const FlatBoxFp& /*prune_box*/,
                             const State& /*state*/) {
        // nothing to do here
    }
};

template <typename OpenIsWorse, typename FocalIsWorse>
struct CountingOutputHeuristic
    : public OutputHeuristic<OpenIsWorse, FocalIsWorse,
                             CountingOutputHeuristic<OpenIsWorse, FocalIsWorse>> {

    using OpenIsWorseT = OpenIsWorse;
    using FocalIsWorseT = FocalIsWorse;
    using SelfT = CountingOutputHeuristic<OpenIsWorse, FocalIsWorse>;
    using BaseT = OutputHeuristic<OpenIsWorse, FocalIsWorse, SelfT>;
    using State = typename BaseT::State;

    std::vector<std::vector<int>> counts;
    int num_solutions = 0;

    CountingOutputHeuristic() : BaseT(), counts{} {}

    void initialize_state(const AddTreeFp& at, OutputState& state) {
        state.gscore = at.base_score(0);
        state.hscore = 0.0;
        state.next_tree = -1;
    }

    void update_gscore(const AddTreeFp& at, OutputState& state,
                       size_t tree_index, NodeId leaf_id) {
        state.gscore += at[tree_index].leaf_value(leaf_id, 0);
        state.fscore += get_count_for(tree_index, leaf_id);
    };

    FloatT get_count_for(size_t tree_index, NodeId leaf_id) const {
        if (tree_index >= counts.size())
            return 1;
        const std::vector<int>& counts_for_tree = counts[tree_index];
        if (counts_for_tree.size() <= static_cast<size_t>(leaf_id))
            return 1;

        FloatT value = 1.0 + static_cast<FloatT>(counts_for_tree[leaf_id]) /
                             static_cast<FloatT>(num_solutions);

        return value;
    }

    void incr_count_for(size_t tree_index, NodeId leaf_id) {
        if (tree_index >= counts.size())
            counts.resize(tree_index+1);
        std::vector<int>& counts_for_tree = counts[tree_index];
        size_t leaf_id_size_t = static_cast<size_t>(leaf_id);
        if (counts_for_tree.size() <= leaf_id_size_t)
            counts_for_tree.resize(leaf_id_size_t+1);
        ++counts_for_tree[leaf_id];
    }

    void notify_new_solution(const AddTreeFp& at,
                             const FlatBoxFp& prune_box,
                             const State& state) {
        this->leafiter.setup_flatbox(state.box, prune_box);
        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index) {
            const auto& t = at[tree_index];
            this->leafiter.setup_tree(t);
            NodeId leaf_id = this->leafiter.next();
            if (leaf_id == -1)
                throw std::runtime_error("leaf_id == -1?");
            if (this->leafiter.next() != -1)
                throw std::runtime_error("not a unique leaf");
            incr_count_for(tree_index, leaf_id);
        }
        ++num_solutions;
    }
};

template <typename OpenIsWorse, typename FocalIsWorse, typename ClassIsWorse>
struct MultiOutputHeuristic {
    using State = OutputState;

    using OpenIsWorseT = OpenIsWorse;
    using FocalIsWorseT = FocalIsWorse;
    using ClassIsWorseT = ClassIsWorse;
    using SelfT = MultiOutputHeuristic<OpenIsWorse, FocalIsWorse, ClassIsWorse>;

    const int num_leaf_values;
    const int num_trees;
    FloatT fail_when_class0_worse_than;

    std::vector<FloatT> buf;
    FloatT *gc;
    FloatT *hc;
    FloatT *hcmin;

    OutputStateOpenIsWorse<OpenIsWorseT> open_isworse;
    FocalIsWorseT focal_isworse;
    ClassIsWorseT class_isworse;

    LeafIter<TreeFp> leafiter;

    MultiOutputHeuristic(const Config& conf, int num_leaf_values,
                         size_t num_trees)
        : num_leaf_values(num_leaf_values)
        , num_trees(static_cast<int>(num_trees))
        , fail_when_class0_worse_than(
                conf.multi_ignore_state_when_class0_worse_than)
        , buf(num_leaf_values*(this->num_trees + 2))
        , open_isworse()
        , focal_isworse()
        , class_isworse()
        , leafiter()
    {
        gc =    &buf[0 * num_leaf_values];
        hc =    &buf[1 * num_leaf_values];
        hcmin = &buf[2 * num_leaf_values];
    }

    bool update_scores(const AddTreeFp& at, const FlatBoxFp& prune_box,
                       State &state) {
        using OrdLimOpen = OrdLimit<FloatT, OpenIsWorse>;
        using OrdLimClass = OrdLimit<FloatT, ClassIsWorse>;

        this->initialize_state(at, state);
        leafiter.setup_flatbox(state.box, prune_box);

        std::fill(hcmin, hcmin + (num_trees * num_leaf_values),
                  OrdLimClass::best(class_isworse));

        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index) {
            FloatT *hcmin_tree = hcmin + (tree_index * num_leaf_values);
            hcmin_tree[0] = OrdLimOpen::worst(open_isworse);

            const TreeFp& tree = at[tree_index];
            leafiter.setup_tree(tree);

            int num_leaves_accessible = 0;
            NodeId leaf_id = leafiter.next();
            for (NodeId i = leaf_id; i != -1; i = leafiter.next()) {
                ++num_leaves_accessible;
                leaf_id = i; // store the last valid leaf_id (avoid -1)
                hcmin_tree[0] = std::max(tree.leaf_value(leaf_id, 0),
                                         hcmin_tree[0], open_isworse);
                for (int c = 1; c < num_leaf_values; ++c) {
                    hcmin_tree[c] = std::min(tree.leaf_value(leaf_id, c),
                                             hcmin_tree[c], class_isworse);
                }
            }

            if (num_leaves_accessible > 1) { // tree's contribution to hc
                for (int c = 0; c < num_leaf_values; ++c)
                    hc[c] += hcmin_tree[c];
            } else { // tree contributes to gscore (unique leaf)
                for (int c = 0; c < num_leaf_values; ++c)
                    gc[c] += tree.leaf_value(leaf_id, c);
                hcmin_tree[0] = std::numeric_limits<FloatT>::quiet_NaN();
                state.fscore += 1;
            }
        }

        int best_c = 1;
        FloatT best_class_score = gc[1] + hc[1];
        for (int c = 2; c < num_leaf_values; ++c) {
            FloatT class_score = gc[c] + hc[c];
            if (class_isworse(best_class_score, class_score)) {
                best_class_score = class_score;
                best_c = c;
            }
        }
        state.gscore = gc[0] - gc[best_c];
        state.hscore = hc[0] - hc[best_c];

        // Set best next tree without considering `best_c`
        FloatT next_tree_best = OrdLimOpen::worst(open_isworse);
        for (size_t tree_index = 0; tree_index < at.size(); ++tree_index) {
            FloatT *hcmin_tree = hcmin + (tree_index * num_leaf_values);
            if (std::isnan(hcmin_tree[0]))
                continue;
            for (int cc = 1; cc < num_leaf_values; ++cc) {
                FloatT next_tree_value = hcmin_tree[0] - hcmin_tree[cc];
                if (open_isworse(next_tree_best, next_tree_value)) {
                    next_tree_best = next_tree_value;
                    state.next_tree = static_cast<int>(tree_index);
                }
            }
        }

        bool g0_good_enough = open_isworse(gc[0] + hc[0],
                fail_when_class0_worse_than);

        // Make this update_scores fail if the heuristic estimate of the output
        // for the first class is not good enough.
        return !g0_good_enough;
    }

    void initialize_state(const AddTreeFp& at, OutputState& state) {
        for (int i = 0; i < num_leaf_values; ++i) {
            gc[i] = at.base_score(i);
            hc[i] = 0.0;
        }
        state.gscore = 0.0;
        state.hscore = 0.0;
        state.next_tree = -1;
    }

    void notify_new_solution(const AddTreeFp& /*at*/,
                             const FlatBoxFp& /*prune_box*/,
                             const State& /*state*/) {
        // nothing to do here
    }
};

using MaxBasicOutputHeuristic = BasicOutputHeuristic<LessIsWorse, LessIsWorse>;
using MinBasicOutputHeuristic = BasicOutputHeuristic<GreaterIsWorse, LessIsWorse>;
using MaxCountingOutputHeuristic = CountingOutputHeuristic<LessIsWorse, LessIsWorse>;
using MinCountingOutputHeuristic = CountingOutputHeuristic<GreaterIsWorse, LessIsWorse>;
using MaxMaxMultiOutputHeuristic = MultiOutputHeuristic<LessIsWorse, LessIsWorse, LessIsWorse>;
using MinMaxMultiOutputHeuristic = MultiOutputHeuristic<GreaterIsWorse, LessIsWorse, LessIsWorse>;
using MaxMinMultiOutputHeuristic = MultiOutputHeuristic<LessIsWorse, LessIsWorse, GreaterIsWorse>;

} // namespace veritas

#endif // VERITAS_HEURISTICS_HPP
