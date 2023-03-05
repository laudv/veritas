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

struct Settings {

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
     */
    FloatT ignore_state_when_worse_than;

    /**
     * Stop when Veritas found a solution with a score that is at least as good
     * as this.
     */
    FloatT stop_when_atleast_bound_better_than;

    /**
     * Constructor taking a comparator for the open state list.
     * We need this for `ignore_state_when_worse_than` and
     * `stop_when_atleast_bound_better_than` because what is _worse_ or _better_
     * is defined by whether we are minimizing or maximizing.
     *
     * No need to call this manually, access `Search::settings` instead.
     */
    template <typename OpenIsWorse>
    inline Settings(const OpenIsWorse& cmp)
        : ignore_state_when_worse_than(
                OrdLimit<FloatT, OpenIsWorse>::worst(cmp))
        , stop_when_atleast_bound_better_than(
                OrdLimit<FloatT, OpenIsWorse>::best(cmp))
    {}
};

struct Statistics {
    size_t num_steps = 0;
    size_t num_states_ignored = 0;
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

using time_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<time_clock>;

class Search {
public:
    Settings settings;
    Statistics stats;

protected:
    AddTree at_;
    AddTreeFp atfp_;
    FpMap fpmap_;
    time_point start_time_;

    size_t max_memory_;
    BlockStore<IntervalPairFp> store_;
    FlatBoxFp prune_box_;

    Search(Settings s, const AddTree& at, const FlatBox& prune_box);

public: // Constructor methods

    static std::shared_ptr<Search> max_output(const AddTree& at,
            const FlatBox& prune_box = {});
    static std::shared_ptr<Search> min_output(const AddTree& at,
            const FlatBox& prune_box = {});

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

    virtual Bounds current_bounds() const = 0;

public:
    double time_since_start() const;

    void set_max_memory(size_t bytes);
    size_t get_max_memory() const;
    size_t get_used_memory() const;

}; // abstract class Search













} // namespace veritas

#endif // VERITAS_FP_SEARCH_HPP
