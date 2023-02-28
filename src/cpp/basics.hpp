/**
 * \file basics.hpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BASICS_HPP
#define VERITAS_BASICS_HPP

#include <iostream>
#include <vector>
#include <limits>

namespace veritas {
    using FloatT = double;
    using FpT = unsigned short; // Fixed point precision type

    using NodeId = int;
    using FeatId = int;


    // SEARCH: https://dirtyhandscoding.wordpress.com/
    //                  2017/08/25/performance-comparison-linear-search-vs-binary-search
    // should also try branchless binary search later
    // `l` is typically a fairly short list of sorted feature ids
    /** branchless linear scan search --> we hope the compiler vectorizes this */
    inline size_t linear_search(const std::vector<FeatId>& l, FeatId x)
    {
        size_t cnt = 0;
        for (const auto& v : l)
            cnt += (v < x);
        return cnt;
    }

    /**
     * A data pointer wrapper. Data is expected to have a
     * [numpy](https://numpy.org/) layout.
     *
     * - https://docs.python.org/3/c-api/buffer.html#buffer-structure
     * - https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
     */
    template <typename T>
    struct data {
        T *ptr;
        size_t num_rows, num_cols;
        size_t stride_row, stride_col; // in num of elems, not bytes

        /** Compute the index of an element. */
        inline size_t index(size_t row, size_t col) const
        { return row * stride_row + col * stride_col; }

        /** Access element in data matrix without bounds checking. */
        inline T get_elem(size_t row, size_t col) const
        { return ptr[index(row, col)]; }

        /** Access elements of first row. */
        inline T operator[](size_t col) const
        { return get_elem(0, col); }

        /** Select a row from the data. */
        inline data row(size_t row) const
        {
            return {
                ptr+index(row, 0),
                1,
                num_cols,
                stride_row,
                stride_col,
            };
        }

        explicit data(std::vector<T>& v)
            : ptr(&v[0])
            , num_rows(1), num_cols(v.size())
            , stride_row(0), stride_col(1) {}

        data(T *ptr, size_t nr, size_t nc, size_t sr, size_t sc)
            : ptr(ptr), num_rows(nr), num_cols(nc), stride_row(sr), stride_col(sc) {}
    };

    template <typename T>
    inline
    std::ostream&
    operator<<(std::ostream& strm, const data<T>& d)
    {
        return strm << "data{ptr=" << d.ptr
            << ", shape=" << d.num_rows << ", " << d.num_cols
            << ", strides=" << d.stride_row << ", " << d.stride_col
            << '}';
    }

    constexpr inline bool check_sanity() {
#ifndef VERITAS_SANITY_CHECKS
        return false;
#elif VERITAS_SANITY_CHECKS == 1
        return true;
#else
        return false;
#endif // !VERITAS_SANITY_CHECKS
    }
}


#endif // VERITAS_BASICS_HPP
