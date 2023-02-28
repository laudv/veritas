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

#include "block_store.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "tree.hpp"
#include "addtree.hpp"

#include <chrono>
#include <functional>
#include <memory>

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
     * In bytes, how much memory can Veritas spend on search states.
     * Default: 1 GiB
     */
    size_t max_memory = size_t(1024)*1024*1024;
};

struct Statistics {
    size_t num_steps = 0;
};

enum class StopReason {
    NONE,
    NO_MORE_OPEN,
    NUM_SOLUTIONS_EXCEEDED,
    NUM_NEW_SOLUTIONS_EXCEEDED,
    OPTIMAL,
    UPPER_LT,
    LOWER_GT,
};

std::ostream& operator<<(std::ostream& strm, StopReason r);

struct Bounds {
    FloatT lower_bound;
    FloatT upper_bound;
};

using time_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<time_clock>;

class Search {
public:
    Settings settings;
    Statistics stats;

protected:
    AddTree at_;
    time_point start_time_;

    inline Search(const AddTree& at)
        : settings{}
        , stats{}
        , at_{at}
        , start_time_{time_clock::now()}
    {}

public: // Constructor methods

    static std::shared_ptr<Search> max_output(const AddTree& at);
    static std::shared_ptr<Search> min_output(const AddTree& at);

public:
    virtual ~Search() { /* required, otherwise pybind11 memory leak */ }

public: // abstract interface methods
    virtual StopReason step() = 0;
    virtual StopReason steps(size_t num_steps) = 0;
    virtual StopReason step_for(double num_seconds, size_t num_steps) = 0;

    virtual size_t num_open() const = 0;
    virtual bool is_optimal() const = 0;
    virtual size_t num_solutions() const = 0;
    //virtual const Solution& get_solution(size_t solution_index) const = 0;

    virtual void set_mem_capacity(size_t bytes) = 0;

    virtual size_t remaining_mem_capacity() const = 0;
    virtual size_t used_mem_size() const = 0;
    virtual Bounds current_bounds() const = 0;

public: // utility
    double time_since_start() const;


}; // abstract class Search













} // namespace veritas

#endif // VERITAS_FP_SEARCH_HPP
