/**
 * \file fp_search.hpp
 *
 * Fixed precision search.
 * Avoid typical floating point issues by reasoning over integers instead of
 * floats.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 */

#ifndef VERITAS_FP_SEARCH_HPP
#define VERITAS_FP_SEARCH_HPP

#include "basics.hpp"
#include "block_store.hpp"
#include "box.hpp"
#include "fp.hpp"
#include "interval.hpp"
#include "tree.hpp"
#include "addtree.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <iostream>

namespace veritas {

enum class HeuristicType {
    MAX_OUTPUT,
    MIN_OUTPUT,
    MAX_COUNTING_OUTPUT,
    MIN_COUNTING_OUTPUT,

    MULTI_MAX_MAX_OUTPUT_DIFF,
    MULTI_MAX_MIN_OUTPUT_DIFF,
    MULTI_MIN_MAX_OUTPUT_DIFF,
};

class Search; // forward decl for Config

struct Config {

    /**
     * The maximum memory the search can use.
     */
    size_t max_memory = size_t(4) * 1024 * 1024 * 1024;

    /**
     * Veritas allocates blocks of memory to store the boxes of the search
     * states with stable pointers. When a block is full, an additional new
     * block twice the size is allocated. This configures the size of the first
     * block.
     *
     * This is mainly useful for testing.
     */
    size_t memory_min_block_size = 5 * 1024 * 1024;

    /**
     * Type of heuristic for the upcoming Search
     */
    HeuristicType heuristic;

    /**
     * Discount factor of state score h in order to qualify for the focal
     * list.
     */
    FloatT focal_eps = 0.8;

    /**
     * Maximum size of of the focal list. The size of the list is limited by
     * this number and by the number of states that are at least as good as
     * `focal_eps` compared to the current optimal.
     */
    size_t max_focal_size = 1000;

    /**
     * Stop Search::steps(..) and Search::step_for(..) when this number of solutions
     * have been found.
     */
    size_t stop_when_num_solutions_exceeds = 9'999'999;

    /**
     * Stop Search::steps(..) and Seach::step_for(..) when this number of _new_
     * solutions have been found, i.e. new solutions since the last call to
     * Search::steps(..) and Seach::step_for(..).
     */
    size_t stop_when_num_new_solutions_exceeds = 9'999'999;

    /**
     * Stop Search::steps(..) and Seach::step_for(..) when the optimal solution
     * has been found. Disable this if you want to find multiple suboptimal
     * solutions.
     */
    bool stop_when_optimal = true;

    /**
     * Ignore search states with bounds that are worse than this.
     *
     * Counted in `stats.num_states_ignored`
     */
    FloatT ignore_state_when_worse_than;

    /**
     * Stop when Veritas finds a solution with a score that is at least as good
     * as this.
     */
    FloatT stop_when_atleast_bound_better_than;

    /**
     * [Multiclass only] Ignore a search state when it has a score for the
     * first action that is worse than the given score.
     *
     * Counted in `stats.num_update_scores_fails`
     */
    FloatT multi_ignore_state_when_class0_worse_than;

    /** Constructor */
    Config(HeuristicType h);

    /**
     * Get a Search instance from this configuration using the configured
     * heuristic.
     */
    std::shared_ptr<Search> get_search(const AddTree& at,
                                       const FlatBox& prune_box) const;

    /**
     * Reuse the heuristic of the given Search. Use this with the counting
     * heuristics to keep the counts between runs.
     */
    std::shared_ptr<Search> reuse_heuristic(const Search& search,
                                            const FlatBox& prune_box) const;
};

struct Statistics {
    size_t num_steps = 0;
    size_t num_states_ignored = 0;
    size_t num_update_scores_fails = 0;
};

enum class StopReason {
    NONE,
    NO_MORE_OPEN,
    NUM_SOLUTIONS_EXCEEDED,
    NUM_NEW_SOLUTIONS_EXCEEDED,
    OPTIMAL,
    ATLEAST_BOUND_BETTER_THAN,
    OUT_OF_TIME,
    OUT_OF_MEMORY,
};

std::ostream& operator<<(std::ostream& strm, StopReason r);

struct Bounds {
    /**
     * The optimal solution is at least as good as this bound. This is simply
     * the score of the best (sub-optimal) solution found so far. For a
     * maximization problem, this is a lower bound on the optimal solution.
     */
    FloatT atleast;

    /**
     * The optimal solution cannot be better than this value.
     * For a maximization problem, this is an upper bound on the optimal solution.
     */
    FloatT best;

    /**
     * The heuristic score of the state at the top of the open queue.
     */
    FloatT top_of_open;

    // Depends on optimization task.
    template <typename OpenIsWorse>
    inline Bounds(const OpenIsWorse& cmp)
        : atleast(
                OrdLimit<FloatT, OpenIsWorse>::best(cmp))
        , best(
                OrdLimit<FloatT, OpenIsWorse>::worst(cmp))
        , top_of_open(
                OrdLimit<FloatT, OpenIsWorse>::worst(cmp))
    {}
};

std::ostream& operator<<(std::ostream& s, const Bounds& bounds);

struct Solution {
    Box::BufT box;
    FloatT output;
    double time;
};

std::ostream& operator<<(std::ostream& s, const Solution& bounds);

using time_clock = std::chrono::system_clock;
using time_point = std::chrono::time_point<time_clock>;

class Search {
public:
    const Config config;
    Statistics stats;

protected:
    AddTree at_;
    AddTreeFp atfp_;
    FpMap fpmap_;
    time_point start_time_;

    BlockStore<IntervalPairFp> store_;
    FlatBoxFp prune_box_;

    Search(const Config& config, const AddTree& at, const FlatBox& prune_box);

public:
    virtual ~Search() { /* required, otherwise pybind11 memory leak */ }

public: // abstract interface methods
    virtual StopReason step() = 0;
    virtual StopReason steps(size_t num_steps) = 0;
    virtual StopReason step_for(double num_seconds, size_t num_steps) = 0;

    virtual size_t num_open() const = 0;
    virtual bool is_optimal() const = 0;
    virtual size_t num_solutions() const = 0;

    virtual Solution get_solution(size_t solution_index) const = 0;
    virtual std::vector<NodeId> get_solution_nodes(size_t solution_index) const = 0;

    virtual Bounds current_bounds() const = 0;

public:
    double time_since_start() const;
    size_t get_used_memory() const;

    const AddTree& get_addtree() const;

}; // abstract class Search













} // namespace veritas

#endif // VERITAS_FP_SEARCH_HPP
