/**
 * \file fp_search.hpp
 *
 * Finite precision search.
 * Avoid typical floating point issues by reasoning over integers instead of
 * floats.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 */

#ifndef VERITAS_FP_SEARCH_HPP
#define VERITAS_FP_SEARCH_HPP

#include <memory>

#include "domain.hpp"
#include "tree.hpp"
#include "block_store.hpp"

namespace veritas {

    struct Settings {

        /**
         * Discount factor of state score h in order to qualify for the focal
         * list.
         */
        FloatT focal_eps;

        /** Default settings. */
        Settings()
            : focal_eps{0.8}
        {}
    };

    struct MaxCmp {
        bool operator()(FloatT a, FloatT b) { return a > b; }
    };

    struct MinCmp {
        bool operator()(FloatT a, FloatT b) { return a < b; }
    };

    /**
     * A state for ensemble output optimization.
     */
    struct OutputState {
        /**
         * Sum of uniquely selected leaf values so far.
         */
        FloatT output;

        /**
         * Overestimate (maximization) or underestimate (minimization) of output
         * that can still be added to g by trees for which multiple leaves are
         * still reachable.
         */
        FloatT heuristic;

        /**
         * Cached focal score, computed by heuristic computation.
         */
        FloatT focal;

        /**
         * Which tree do we merge into this state next? This is determined by
         * the heuristic computation.
         */
        int next_tree;

        /**
         * Scoring function for this state for the open list.
         */
        FloatT open_score() const {
            return output + heuristic;
        }

        /**
         * Scoring function for this state for the focal list.
         */
        FloatT focal_score() const {
            return focal;
        }
    };

    namespace heuristic_detail {

        FloatT compute_basic_output_heuristic();

    } // namespace heuristic_detail

    template <typename OpenCmp, typename FocalCmp>
    struct BasicOutputHeuristic {
        using State = OutputState;

        OpenCmp open_cmp;
        FocalCmp focal_cmp;

        BasicOutputHeuristic() : open_cmp{}, focal_cmp{} {}

        void update(State& out)
        {

        }
    };

    template <typename OpenCmp, typename FocalCmp>
    struct CountingOutputHeuristic
        : public BasicOutputHeuristic<OpenCmp, FocalCmp> {

    };

    template <typename Heuristic>
    class SearchImpl {
    public:
        using State = typename Heuristic::State;

        Settings settings;

    private:
        std::shared_ptr<Heuristic> h_;

    public:
        SearchImpl(std::shared_ptr<Heuristic> h, Settings s = {}) // takes shared ownership of heuristic
            : settings{s}
            , h_{std::move(h)}
        {}
    };

} // namespace veritas

#endif // VERITAS_FP_SEARCH_HPP
