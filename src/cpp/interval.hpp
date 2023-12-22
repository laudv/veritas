/**
 * \file interval.hpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_INTERVAL_HPP
#define VERITAS_INTERVAL_HPP

#include "basics.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace veritas {

template <typename T> struct GInterval { // generic interval
    using ValueT = T;

    T lo; // inclusive
    T hi; // exclusive

    GInterval() : GInterval{Limits<T>::min, Limits<T>::max} {}
    GInterval(T lo, T hi) : lo{lo}, hi{hi} {
        if constexpr (check_sanity()) {
            GInterval::check_or_throw(lo, hi);
        }
    }

    static inline void check_or_throw(T lo, T hi) {
        if (lo >= hi) {
            std::stringstream s;
            s << "Interval<"
              << typeid(T).name()
              << "> error: lo >= hi: [" << lo << ", " << hi << "]";
            throw std::invalid_argument(s.str());
        }
    }

    static inline GInterval checked(T lo, T hi) {
        check_or_throw(lo, hi);
        return { lo, hi };
    }

    static inline GInterval from_lo(T lo) { return {lo, Limits<T>::max}; }
    static inline GInterval from_hi(T hi) { return {Limits<T>::min, hi}; }
    static inline GInterval constant(T x) { return {x, std::nextafter(x, Limits<T>::max)}; }

    inline bool lo_is_unbound() const { return lo == Limits<T>::min; }
    inline bool hi_is_unbound() const { return hi == Limits<T>::max; }
    inline bool operator==(const GInterval& o) const { return o.lo == lo && o.hi == hi; }
    inline bool operator!=(const GInterval& o) const { return !(*this == o); }
    inline bool is_everything() const { return *this == GInterval(); }
    inline bool contains(T v) { return lo <= v && v < hi; }
    inline bool overlaps(const GInterval& o) const {
        // [     )
        //       [    )    hi is exclusive
        return lo < o.hi && hi > o.lo;
    }

    inline GInterval intersect(const GInterval& o) const {
        return { std::max(lo, o.lo), std::min(hi, o.hi) };
    }

    inline std::tuple<GInterval<T>, GInterval<T>> split(T value) const {
        return { {lo, value}, {value, hi} };
    }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &s, const GInterval<T> &d) {
     if (d.is_everything())
         return s << "Interval()";
     if (d.hi_is_unbound())
         return s << "Interval(>=" << d.lo << ')';
     if (d.lo_is_unbound())
         return s << "Interval(<" << d.hi << ')';
    return s << "Interval(" << d.lo << ',' << d.hi << ')';
}

using IntervalFp = GInterval<FpT>;
using Interval = GInterval<FloatT>;

/** Split value used for boolean splits (assuming feature values in {0, 1}) */
const FloatT BOOL_SPLIT_VALUE = 0.5;
/** (-inf, 0.5) interval for FALSE */
const Interval FALSE_INTERVAL{Limits<FloatT>::min, BOOL_SPLIT_VALUE};
/** [0.5, inf) domain for TRUE */
const Interval TRUE_INTERVAL{BOOL_SPLIT_VALUE, Limits<FloatT>::max};

template <typename T>
struct GLtSplit { // generic LtSplit
    using IntervalT = GInterval<T>;
    using ValueT = T;

    FeatId feat_id;
    ValueT split_value;

    inline GLtSplit() : feat_id{}, split_value{} {}
    inline GLtSplit(FeatId f, ValueT v) : feat_id{f}, split_value{v} {}

    /** True goes left, false goes right */
    inline bool test(T v) const { return v < split_value; }

    /** Evaluate this split on an instance (FloatT only). */
    bool test(const data<T>& row) const {
        return test(row[feat_id]);
    }

    /** Get the left and right domains of this split.
     * Strict less than, so eq goes right */
    inline std::tuple<IntervalT, IntervalT> get_intervals() const
    { return IntervalT().split(split_value); }

    inline bool operator==(const GLtSplit& o) const
    { return feat_id == o.feat_id && split_value == o.split_value; }

    inline bool operator!=(const GLtSplit& o) const { return !(*this == o); }
};

template <typename T>
std::ostream &operator<<(std::ostream &strm, const GLtSplit<T> &s) {
    return strm << "F" << s.feat_id << " < " << s.split_value;
}

using LtSplit = GLtSplit<FloatT>;
using LtSplitFp = GLtSplit<FpT>;

/** A boolean < 0.5 split for features with values {0.0, 1.0} */
inline LtSplit bool_ltsplit(FeatId feat_id) {
    return LtSplit(feat_id, BOOL_SPLIT_VALUE);
}






} // namespace veritas
#endif // VERITAS_INTERVAL_HPP
