/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BASICS_HPP
#define VERITAS_BASICS_HPP

#include <cstdint>
#include <vector>
#include <limits>

namespace veritas {
    using FloatT = float;

    static const FloatT FLOATT_INF = std::numeric_limits<FloatT>::infinity();

    using TreeId = int;
    using NodeId = int;
    using FeatId = int;


    // SEARCH: https://dirtyhandscoding.wordpress.com/
    //                  2017/08/25/performance-comparison-linear-search-vs-binary-search
    // should also try branchless binary search later
    // `l` is typically a fairly short list of sorted feature ids
    /** branchless linear scan search --> we hope the compiler vectorizes this */
    inline size_t linear_search(const std::vector<FeatId> l, FeatId x)
    {
        size_t cnt = 0;
        for (const auto& v : l)
            cnt += (v < x);
        return cnt;
    }


}


#endif // VERITAS_BASICS_HPP
