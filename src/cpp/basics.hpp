/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BASICS_HPP
#define VERITAS_BASICS_HPP

#include <iostream>
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

    struct data {
        FloatT *ptr;
        size_t num_rows, num_cols;
    };

    struct row_major_data : public data {
        inline FloatT get_elem(size_t row, size_t col) const {
            FloatT v = ptr[row * num_cols + col];
            return v;
        }
    };

    struct col_major_data : public data {
        inline FloatT get_elem(size_t row, size_t col) const {
            FloatT v = ptr[col * num_rows + row];
            return v;
        }
    };

    template <typename D>
    struct row {
        D data;
        size_t row;
        inline FloatT operator[](size_t col) const {
            return data.get_elem(row, col);
        }
    };


}


#endif // VERITAS_BASICS_HPP
